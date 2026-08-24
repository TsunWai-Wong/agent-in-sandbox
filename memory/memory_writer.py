"""What happens to a conversation after a run ends.

One pipeline, in three steps, and it used to be three files:

    when      enough turns, or turns old enough
    extract   one structured call: transcript -> drafts
    record    per draft: retrieve what might cover it, pick a verb, write

The steps stay separate in the reading and merged in the file because there is
exactly one path through them and exactly one caller. What was a Reconciler
class existed to be constructed once and called from here.

The division that does matter is kept: this proposes and judges, `memory_store.py`
writes. And a draft never carries a verb — whether a fact is new or a revision
is settled against rows the extractor never saw.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor

from agent.event_store import USER_MESSAGE, RunState
from agent.middleware import CONTINUE, Decision, Middleware
from agent.prompts import Prompt
from monitoring import get_tracer

from .memory_store import DEFAULT_NEIGHBOURS, NEIGHBOUR_THRESHOLD, UserMemoryStore
from .types import INFERRED, MemoryContent, MemoryContents, MemoryOperation


logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)

# Enough turns to be worth an extraction, low enough that an abandoned
# conversation loses a little rather than all of it.
FLUSH_EVERY_TURNS = 6

# The other half: turns that never reach the count still get written once they
# are this old. A session has no end, so "wait for close" would wait forever.
MAX_AGE_SECONDS = 900.0

MAX_EVENTS_PER_FLUSH = 24
MAX_TEXT_CHARS = 2_000
MAX_CANDIDATES = 6


class MemoryWriter(Middleware):
    """Extracts and records what is worth remembering, off the answer's path.

    Holds no run state: what is unwritten is a watermark on the event store, so
    two flushes cannot dispatch the same turns and a restart does not re-read
    what it already recorded.
    """

    def __init__(
        self,
        store: UserMemoryStore,
        llm,
        user_id: str,
        flush_every: int = FLUSH_EVERY_TURNS,
        max_age: float = MAX_AGE_SECONDS,
        enabled: bool = True,
    ) -> None:
        self.store = store
        self.llm = llm
        self.user_id = user_id
        self.flush_every = flush_every
        self.max_age = max_age
        self.enabled = enabled
        # One worker, so extractions queue instead of racing into the store.
        # "Async" means the answer did not wait, not that it may be killed
        # halfway — see join().
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="memory")

    # -- when ----------------------------------------------------------------

    def on_run_end(self, state: RunState, answer: str) -> Decision:
        if not self.enabled:
            # Recorded, not skipped quietly: a disabled writer and a broken one
            # are otherwise indistinguishable, and "why did it forget?" is the
            # question you actually have later.
            state.append("memory", "memory_skipped", key="disabled")
            return CONTINUE
        reason = self._due(state)
        if reason:
            self.flush(state, reason)
        return CONTINUE

    def _due(self, state: RunState) -> str | None:
        """Count catches a busy conversation, age catches a quiet one.

        Age is checked on the way out of the *next* run, so turns left behind by
        someone who walked away are written when they come back — the first
        moment anything is running to notice.
        """
        events = state.event_store.unrecorded_turns()
        if not events:
            return None
        if sum(1 for e in events if e.kind == USER_MESSAGE) >= self.flush_every:
            return "turn_count"
        if time.time() - events[0].at >= self.max_age:
            return "age"
        return None

    def flush(self, state: RunState, reason: str = "manual") -> None:
        """Dispatch an extraction and move the watermark.

        Nothing about remembering may take a run down — the answer the user
        already has is correct without any of it. What it must not do is fail
        silently, which is why the dispatch lands on the event store.
        """
        events = state.event_store.unrecorded_turns()
        if not events:
            return
        # Stamped before the dispatch: extraction runs on another thread, and a
        # watermark written when it finishes would let the next flush re-send
        # everything it is still working on.
        state.append(
            "memory",
            "memory_flushed",
            key=reason,
            through_seq=events[-1].seq,
            turns=len(events),
        )
        self._pool.submit(self._run, render_transcript(events), state.run_id)

    def join(self, timeout: float = 10.0) -> None:
        """A flush nobody waits for does not happen — the process exits first."""
        self._pool.shutdown(wait=True)

    # -- extract -------------------------------------------------------------

    def _run(self, transcript: str, run_id: str) -> None:
        for draft in self._extract(transcript):
            try:
                self._record(draft, run_id)
            except Exception:
                # One bad draft must not cost the rest of the batch.
                logger.exception("could not record a memory; continuing")

    def _extract(self, transcript: str) -> list[MemoryContent]:
        """One call, every draft at once. Structured output rather than a tool
        loop: there is nothing to look up and no tool to choose."""
        try:
            response = self.llm.chat(
                messages=[self.llm.user_message(transcript)],
                system=Prompt.get_extract_instruction(),
                text_format=MemoryContents,
            )
        except Exception:
            logger.exception("memory extraction failed; nothing recorded")
            return []
        parsed = response.parsed
        return parsed.memories if isinstance(parsed, MemoryContents) else []

    # -- record --------------------------------------------------------------

    def _record(self, draft: MemoryContent, run_id: str) -> dict | None:
        """Reconcile one draft against what is stored, and write it.

        The only path that writes. Whatever the extractor proposed, the store
        sees a checked verb against a row this user owns.
        """
        with tracer.start_as_current_span(
            "memory_reconcile", openinference_span_kind="chain"
        ) as span:
            span.set_input(draft.content)

            # Only rows close enough to be the same fact: a verdict may act on
            # nothing else, so a hallucinated or borrowed id cannot reach the
            # store. The cost is that a rephrasing sharing no vocabulary with
            # what it supersedes will not come back, and is added alongside.
            candidates = self.store.search(
                self.user_id, draft.content, DEFAULT_NEIGHBOURS, NEIGHBOUR_THRESHOLD
            )[:MAX_CANDIDATES]
            verdict = self._decide(draft, candidates)

            span.set_attribute("memory.candidates", len(candidates))
            span.set_attribute("memory.verb", verdict.verb)
            span.set_output(verdict.reason)
            return self._write(draft, verdict, run_id)

    def _decide(self, draft: MemoryContent, candidates: list[dict]) -> MemoryOperation:
        """Pick a verb, spending a model call only when one is warranted."""
        if not candidates:
            return MemoryOperation(verb="add", reason="nothing on file covers this")

        for row in candidates:
            if _normalise(row.get("content")) == _normalise(draft.content):
                return MemoryOperation(
                    verb="noop", target_id=row["id"], reason="already stored, word for word"
                )

        return self._enforce(draft, candidates, self._ask(draft, candidates))

    def _ask(self, draft: MemoryContent, candidates: list[dict]) -> MemoryOperation:
        """The ambiguous case, and the only one that costs anything."""
        payload = "\n".join(
            [
                "[Proposed memory]",
                _as_line(draft.model_dump(mode="json")),
                "",
                "[Existing memories]",
                *(_as_line(row) for row in candidates),
            ]
        )
        try:
            response = self.llm.chat(
                messages=[self.llm.user_message(payload)],
                system=Prompt.get_reconcile_instruction(),
                text_format=MemoryOperation,
            )
        except Exception as error:
            return _add_anyway(f"model call failed: {error}")
        if not isinstance(response.parsed, MemoryOperation):
            return _add_anyway("model returned no verdict")
        return response.parsed

    def _enforce(
        self, draft: MemoryContent, candidates: list[dict], verdict: MemoryOperation
    ) -> MemoryOperation:
        """Check the model's answer against rules it does not get a say in."""
        if verdict.verb not in ("update", "invalidate"):
            return verdict

        target = {row["id"]: row for row in candidates}.get(verdict.target_id)
        if target is None:
            # A hallucinated id and someone else's id both land here: both are
            # simply absent from what was shown.
            return _add_anyway(f"target {verdict.target_id} was not offered")

        # A guess does not overwrite something the user said outright. It is
        # still recorded on its own if it adds anything.
        if draft.confidence <= INFERRED and target["confidence"] > draft.confidence:
            return MemoryOperation(
                verb="noop",
                target_id=target["id"],
                reason="an inferred guess does not supersede a stated memory",
            )
        return verdict

    def _write(
        self, draft: MemoryContent, verdict: MemoryOperation, run_id: str
    ) -> dict | None:
        payload = draft.model_dump(mode="json") | {"source_run_id": run_id}
        try:
            if verdict.verb == "add":
                return self.store.add(self.user_id, payload)
            if verdict.verb == "update":
                return self.store.update(
                    self.user_id,
                    verdict.target_id,
                    {
                        "content": draft.content,
                        "memory_type": draft.memory_type.value,
                        "confidence": draft.confidence,
                    },
                )
            if verdict.verb == "invalidate":
                return self.store.delete(self.user_id, verdict.target_id, run_id)
        except KeyError as error:
            # The target went inactive between the read and the write. Rare, and
            # not worth a transaction spanning a model call.
            logger.warning("Reconcile target vanished (%s); adding instead", error)
            return self.store.add(self.user_id, payload)
        return None


# -- rendering ---------------------------------------------------------------


def render_transcript(events) -> str:
    """Flatten unwritten events for the extractor.

    Reads `text`, which is why the store keeps it beside the payload — this must
    not parse a provider structure to find out what was said.

    Both halves: the user's own words are where stated preferences live, but an
    answer is often the only thing that makes one mean anything. "Yes, do it
    that way" has no content on its own.
    """
    labels = {"user_message": "User", "agent_message": "Assistant"}
    blocks = []
    for event in events[-MAX_EVENTS_PER_FLUSH:]:
        text = event.text
        if len(text) > MAX_TEXT_CHARS:
            text = text[:MAX_TEXT_CHARS] + "\n…[truncated]"
        blocks.append(f"{labels.get(event.kind, event.kind)}: {text}")
    return "\n\n".join(blocks)


def _as_line(memory: dict) -> str:
    """One draft or one stored row, for the reconcile prompt. The id leads on
    candidates: it is the handle a verdict acts through."""
    head = f"[id={memory['id']}] " if memory.get("id") else ""
    when = f" recorded={memory['created_at']:%Y-%m-%d}" if memory.get("created_at") else ""
    return (
        f"{head}type={memory['memory_type']} "
        f"confidence={memory['confidence']}{when}\n  {memory['content']}"
    )


def _normalise(text: str | None) -> str:
    return " ".join((text or "").lower().split())


def _add_anyway(why: str) -> MemoryOperation:
    """When the judgment cannot be made, keep the information. A spare row is
    visible and can be superseded; a noop throws the memory away silently."""
    logger.warning("Reconcile falling back (%s)", why)
    return MemoryOperation(verb="add", reason=f"fallback: {why}")
