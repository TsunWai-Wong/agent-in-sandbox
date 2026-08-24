"""Context management as a guardrail, which is what it always was.

On Conversation it could only run after an answer came back, so one run that
went ten turns deep on fat tool output blew the budget mid-flight — and callers
that skipped Conversation (sub-agents, one-shot ask()) got none of it at all.
At before_model it fires on every model call in every run.

It needs no new seam and no new decision: messages are rebuilt from the log each
call, so appending one event is enough. It still says CONTINUE, like any
middleware with nothing to object to.

Policy only. How to shrink history is ContextAssembler's — this decides when,
and owns the one part that costs money: the summarization call.
"""

from __future__ import annotations

import logging

from .context_assembler import split_at_turn_boundary
from .event_store import HISTORY_SUMMARIZED, RunState
from .middleware import CONTINUE, Decision, Middleware
from .prompts import Prompt


logger = logging.getLogger(__name__)

# A cost ceiling, not the model's context limit — budgeting to the window would
# mean never compacting until a question had already cost a fortune.
DEFAULT_TOKEN_BUDGET = 60_000

# One threshold, not two. Evicting tool traffic from older turns is
# deterministic and free, so the assembler simply always does it. Only
# summarizing costs a model call, and only what costs money needs a record.
COMPRESS_THRESHOLD = 0.7

KEEP_LAST_TURNS = 3


class ContextBudget(Middleware):
    """Summarizes the older half of the conversation when it gets expensive.

    Holds configuration and an LLMService, no run state: the watermark saying
    what is already summarized is on the log, which is what stops it paying for
    the same turns twice.
    """

    def __init__(
        self,
        llm,
        token_budget: int = DEFAULT_TOKEN_BUDGET,
        compress_threshold: float = COMPRESS_THRESHOLD,
        keep_last_turns: int = KEEP_LAST_TURNS,
        summary_model: str | None = None,
    ):
        self.llm = llm
        self.token_budget = token_budget
        self.compress_threshold = compress_threshold
        self.keep_last_turns = keep_last_turns
        self.summary_model = summary_model

    def before_model(self, state: RunState) -> Decision:
        # What the last call read. Lags by one — nothing knows a request's size
        # before making it — so the per-tool output cap bounds the overshoot.
        call = state.last("model_call")
        used = int(call.data.get("input_tokens", 0)) if call else 0
        if used < self.token_budget * self.compress_threshold:
            return CONTINUE

        history = state.session_history()
        older, _ = split_at_turn_boundary(history, self.keep_last_turns)

        # history_events() returns the last summary followed by only what it did
        # not cover, so anything in `older` beyond that summary is unsummarized
        # by definition. Nothing beyond it means one turn is over budget on its
        # own, which summarizing cannot fix.
        unsummarized = [e for e in older if e.kind != HISTORY_SUMMARIZED]
        if not unsummarized:
            return CONTINUE

        summary = self._summarize(older)
        if summary is None:
            # A long history costs tokens; a lost one costs the conversation.
            # Nothing was appended, so the next call simply tries again.
            return CONTINUE

        state.append(
            "before_model",
            HISTORY_SUMMARIZED,
            text=summary,
            covering=len(older),
        )
        logger.info("Compacted at %d tokens: summarized %d events", used, len(older))
        return CONTINUE

    def _summarize(self, events) -> str | None:
        """One cheap call over the text of the older turns.

        Reads `text`, never `payload`: the summarizer wants what was said, not
        tokens spent on tool-call ids.
        """
        transcript = "\n\n".join(f"{e.kind}: {e.text}" for e in events if e.text)
        if not transcript:
            return None
        try:
            response = self.llm.chat(
                messages=[self.llm.user_message(transcript)],
                system=Prompt.get_summary_instruction(),
                model=self.summary_model,
            )
        except Exception:
            logger.exception("Summarizing the conversation failed; keeping history.")
            return None
        return response.text or None
