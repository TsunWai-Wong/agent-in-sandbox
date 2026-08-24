"""The record of a session: an append-only EventStore, and the RunState lens on it.

The store belongs to a session — every run appends to the same log, sub-agents
included. RunState is that log filtered to one run_id, which is what keeps
per-run questions answerable. Get that wrong and AttemptBudget reads a
session-wide count and trips on the fourth question of a conversation.

Nothing here calls anything. Middleware read this and return a decision.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Iterator


USER_MESSAGE = "user_message"
AGENT_MESSAGE = "agent_message"
TOOL_RESULT = "tool_result"
HISTORY_SUMMARIZED = "history_summarized"

# Which kinds are messages rather than bookkeeping. Explicit, so a new
# bookkeeping kind cannot accidentally start appearing in the model's context.
HISTORY_KINDS = frozenset(
    {USER_MESSAGE, AGENT_MESSAGE, TOOL_RESULT, HISTORY_SUMMARIZED}
)

# What memory may read. The agent's turn is here only as the referent an answer
# like "yes, do it that way" needs; the extraction prompt says which half to mine.
MEMORY_KINDS = frozenset({USER_MESSAGE, AGENT_MESSAGE})


@dataclass(frozen=True)
class Event:
    """One line in the store.

    `key` is what the event was about — an artifact hash, a tool-call
    fingerprint, a sub-agent name. Binding an event to its subject is what stops
    an approval earned for one thing being spent on another.

    `text` and `payload` are the two halves of a message and both are needed.
    `text` is what a summarizer, memory or a transcript reads. `payload` is the
    same turn in the provider's own shape, because text cannot be replayed: it
    loses tool-call ids, batch pairing, image parts, reasoning blocks. Must stay
    JSON-serializable, or the run cannot resume in another process.

    `depth` is denormalized from the run because assembly filters on it every
    single model call, to keep a reviewer's turns out of the user's history.
    """

    seq: int
    run_id: str
    seam: str
    kind: str
    key: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    text: str = ""
    payload: Any | None = None
    depth: int = 0
    at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict) -> "Event":
        return cls(**raw)


class EventStore:
    """Append-only log of everything that happened in a session.

    No update, no delete. A gate that passed and was later invalidated leaves
    both lines behind, which is the difference between "the reviewer approved
    this" and "the reviewer approved something, once".

    Every reader takes an optional run_id. None means the whole session, which
    is right for a session total and wrong for anything a middleware asks — so
    middleware are handed a RunState, never this.
    """

    def __init__(self, events: list[Event] | None = None) -> None:
        self._events: list[Event] = list(events or [])

    def append(
        self,
        run_id: str,
        seam: str,
        kind: str,
        key: str = "",
        *,
        text: str = "",
        payload: Any | None = None,
        depth: int = 0,
        **data,
    ) -> Event:
        # seq is assigned here only in memory. A durable store lets the database
        # assign it under UNIQUE (session_id, seq), which is what stops two
        # concurrent runs in one session from claiming the same number.
        event = Event(
            seq=len(self._events),
            run_id=run_id,
            seam=seam,
            kind=kind,
            key=key,
            data=data,
            text=text,
            payload=payload,
            depth=depth,
        )
        self._events.append(event)
        return event

    def all(
        self,
        kind: str | None = None,
        key: str | None = None,
        run_id: str | None = None,
        since: int = -1,
    ) -> list[Event]:
        return [
            e
            for e in self._events
            if e.seq > since
            and (kind is None or e.kind == kind)
            and (key is None or e.key == key)
            and (run_id is None or e.run_id == run_id)
        ]

    def count(self, kind: str, key: str | None = None, run_id: str | None = None) -> int:
        return len(self.all(kind, key, run_id))

    def has(self, kind: str, key: str | None = None, run_id: str | None = None) -> bool:
        return bool(self.all(kind, key, run_id))

    def last(
        self, kind: str, key: str | None = None, run_id: str | None = None
    ) -> Event | None:
        found = self.all(kind, key, run_id)
        return found[-1] if found else None

    def sum(self, kind: str, field_name: str, run_id: str | None = None) -> int:
        """Total one numeric field. Token spend lives here rather than in a
        counter, so a sub-agent's cost lands in the same place as the parent's."""
        return sum(int(e.data.get(field_name, 0)) for e in self.all(kind, None, run_id))

    # -- session-wide views, deliberately unfiltered by run --------------------

    def history_events(self) -> list[Event]:
        """The events a request's message list is built from.

        Everything before the last summary is represented by that summary, so it
        leads the list and the turns it covered are dropped. Returns events, not
        messages: shaping them for a provider is the assembler's job.
        """
        summary = self.last(HISTORY_SUMMARIZED)
        since = summary.seq - 1 if summary else -1
        return [
            e
            for e in self.all(since=since)
            if e.depth == 0 and e.kind in HISTORY_KINDS
        ]

    def unrecorded_turns(self) -> list[Event]:
        """Conversation events the memory writer has not seen yet.

        One watermark replaces the old three triggers, which existed only
        because compaction destroyed turns. Nothing to race now: if extraction
        fails, the turns are still here and it runs again.
        """
        flushed = self.last("memory_flushed")
        since = int(flushed.data.get("through_seq", -1)) if flushed else -1
        return [
            e for e in self.all(since=since) if e.depth == 0 and e.kind in MEMORY_KINDS
        ]

    def __iter__(self) -> Iterator[Event]:
        return iter(self._events)

    def __len__(self) -> int:
        return len(self._events)

    def to_dict(self) -> dict:
        return {"events": [e.to_dict() for e in self._events]}

    @classmethod
    def from_dict(cls, raw: dict) -> "EventStore":
        return cls([Event.from_dict(e) for e in raw["events"]])


@dataclass
class RunState:
    """One run: its identity, and its view of the session's event store.

    The store is a reference to the session's log, not a private one. The run
    can still ask about only itself because every reader below binds run_id.

    Holds references and hashes, never contents. The old history_ref is gone —
    history is events now, so there is nothing left to point at.
    """

    run_id: str
    task: str
    depth: int = 0
    parent_run_id: str | None = None
    workspace_ref: str = ""
    event_store: EventStore = field(default_factory=EventStore)

    @classmethod
    def begin(
        cls,
        task: str,
        event_store: EventStore | None = None,
        depth: int = 0,
        parent_run_id: str | None = None,
    ) -> "RunState":
        # event_store=None builds its own, so a caller with no session still
        # works: eval scripts, tests with a hand-built state, and the agent
        # loop's own fallback when none is offered.
        state = cls(
            run_id=uuid.uuid4().hex[:12],
            task=task,
            depth=depth,
            parent_run_id=parent_run_id,
            event_store=event_store if event_store is not None else EventStore(),
        )
        # No depth= in the data: it is a field on every event now, and passing it
        # here would collide with the one append() sets. See the reserved names.
        state.append("run", "run_started", key=state.run_id, task=task)
        return state

    def child(self, task: str) -> "RunState":
        """A sub-agent's state, one level deeper, on the same store.

        Sharing the log makes the session trace complete. It costs no isolation:
        the run_id is its own, so gates, attempts and budget stay separate.
        """
        return RunState.begin(
            task,
            event_store=self.event_store,
            depth=self.depth + 1,
            parent_run_id=self.run_id,
        )

    # -- the store, forwarded with run_id bound --------------------------------

    def append(
        self,
        seam: str,
        kind: str,
        key: str = "",
        *,
        text: str = "",
        payload: Any | None = None,
        **data,
    ) -> Event:
        """Reserved names, because **data cannot shadow an Event field:
        run_id, seam, kind, key, text, payload, depth, seq, at."""
        return self.event_store.append(
            self.run_id, seam, kind, key,
            text=text, payload=payload, depth=self.depth, **data,
        )

    def count(self, kind: str, key: str | None = None) -> int:
        return self.event_store.count(kind, key, self.run_id)

    def has(self, kind: str, key: str | None = None) -> bool:
        return self.event_store.has(kind, key, self.run_id)

    def last(self, kind: str, key: str | None = None) -> Event | None:
        return self.event_store.last(kind, key, self.run_id)

    def all(self, kind: str | None = None, key: str | None = None) -> list[Event]:
        return self.event_store.all(kind, key, self.run_id)

    # The one deliberate exception, named so it cannot be taken by accident.
    # Context belongs to the conversation, not the run: a budget that reset every
    # question would let a long conversation past a limit it already crossed.
    def session_history(self) -> list[Event]:
        return self.event_store.history_events()

    def record_message(self, kind: str, message, key: str = "") -> Event:
        """Record a history-bearing turn.

        Takes the Message and fills both halves itself, so `text` and `payload`
        cannot drift: one is a projection of the other, written once. `text` is
        stored alongside rather than derived on read because memory and the
        summarizer query it constantly and must not parse a payload to do it.
        """
        return self.append(
            "history", kind, key, text=message.text or "", payload=message.model_dump()
        )

    # -- derived ---------------------------------------------------------------

    @property
    def artifact_hash(self) -> str:
        event = self.last("artifact_hashed")
        return event.key if event else ""

    @property
    def attempts(self) -> int:
        return self.count("model_call")

    @property
    def total_tokens(self) -> int:
        """This run's spend, sub-agents included. Both halves filter to this run
        and both stay correct: the parent is what appends subagent_returned, so
        the rollup carries the parent's run_id."""
        return self.event_store.sum(
            "model_call", "tokens", self.run_id
        ) + self.event_store.sum("subagent_returned", "tokens", self.run_id)

    def stamp_artifact(self, digest: str) -> Event:
        """Gate results bind to the hash they were checked against, so a new
        hash strands every approval earned against the old one."""
        return self.append("artifact", "artifact_hashed", key=digest)

    # The store is not serialized here — it belongs to the session, and writing
    # it per run would give you as many copies as there were turns.
    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "task": self.task,
            "depth": self.depth,
            "parent_run_id": self.parent_run_id,
            "workspace_ref": self.workspace_ref,
        }

    @classmethod
    def from_dict(cls, raw: dict, event_store: EventStore) -> "RunState":
        return cls(event_store=event_store, **raw)
