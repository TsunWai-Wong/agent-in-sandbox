"""The record of one run: an append-only Ledger, and the RunState read from it.

The split is the whole design. The Ledger only ever grows — a step happens, a
line is written, nothing is edited and nothing is removed. RunState is the
current view over those lines plus the handful of identifiers that never change.

Nothing in here calls anything. Middleware read this and return a decision; the
agent loop is the only caller. That is what keeps every interruption point
visible by reading the loop top to bottom, and what lets a run be serialized
mid-flight and picked up in another process.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Iterator


@dataclass(frozen=True)
class Event:
    """One line in the ledger.

    `key` is what the event was *about* — an artifact hash, a tool-call
    fingerprint, a sub-agent name. Binding an event to its subject is what makes
    "was this exact call approved?" answerable, and what stops an approval
    earned for one thing from being spent on another.
    """

    seq: int
    seam: str
    kind: str
    key: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "seq": self.seq,
            "seam": self.seam,
            "kind": self.kind,
            "key": self.key,
            "data": self.data,
            "at": self.at,
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "Event":
        return cls(**raw)


class Ledger:
    """Append-only log of everything that happened in a run.

    No update, no delete. A gate that passed and was later invalidated leaves
    both lines behind, which is the difference between "the reviewer approved
    this" and "the reviewer approved *something*, once".
    """

    def __init__(self, events: list[Event] | None = None) -> None:
        self._events: list[Event] = list(events or [])

    def append(self, seam: str, kind: str, key: str = "", **data) -> Event:
        event = Event(len(self._events), seam, kind, key, data)
        self._events.append(event)
        return event

    def all(self, kind: str | None = None, key: str | None = None) -> list[Event]:
        return [
            e
            for e in self._events
            if (kind is None or e.kind == kind) and (key is None or e.key == key)
        ]

    def count(self, kind: str, key: str | None = None) -> int:
        return len(self.all(kind, key))

    def has(self, kind: str, key: str | None = None) -> bool:
        return any(
            (kind == e.kind) and (key is None or e.key == key) for e in self._events
        )

    def last(self, kind: str, key: str | None = None) -> Event | None:
        for event in reversed(self._events):
            if event.kind == kind and (key is None or event.key == key):
                return event
        return None

    def sum(self, kind: str, field_name: str) -> int:
        """Total one numeric field across every event of a kind.

        Token spend lives here rather than in a counter so that a sub-agent's
        cost lands in the same place as the main agent's, and so no middleware
        has to carry a running total of its own.
        """
        return sum(int(e.data.get(field_name, 0)) for e in self.all(kind))

    def __len__(self) -> int:
        return len(self._events)

    def __iter__(self) -> Iterator[Event]:
        return iter(self._events)

    def to_dict(self) -> dict:
        return {"events": [e.to_dict() for e in self._events]}

    @classmethod
    def from_dict(cls, raw: dict) -> "Ledger":
        return cls([Event.from_dict(e) for e in raw["events"]])


@dataclass
class RunState:
    """One agentic loop run: its identity, and the current view of its ledger.

    Six stored fields and a log. Everything a middleware asks — how many turns
    have we taken, was this approved, has review passed on this exact answer —
    is derived, so middleware hold no counters of their own. That is what makes
    them stateless, testable against a hand-built ledger, and able to survive a
    pause that outlives the process.

    What is deliberately NOT here: the draft, the message history, the tool
    outputs. Each of those already has exactly one home, and copying one in
    would give you two that drift. State holds references and hashes, never
    contents. The test for any field you are tempted to add: something must
    branch on it. If nothing does, it is an event, not a field.
    """

    run_id: str
    task: str
    depth: int = 0
    parent_run_id: str | None = None
    # References, never contents — so a resume can rehydrate the workspace and
    # the history alongside the state instead of hoping the caller kept them.
    workspace_ref: str = ""
    history_ref: str = ""
    ledger: Ledger = field(default_factory=Ledger)

    @classmethod
    def begin(
        cls, task: str, depth: int = 0, parent_run_id: str | None = None
    ) -> "RunState":
        state = cls(
            run_id=uuid.uuid4().hex[:12],
            task=task,
            depth=depth,
            parent_run_id=parent_run_id,
        )
        state.append("run", "run_started", key=state.run_id, task=task, depth=depth)
        return state

    def child(self, task: str) -> "RunState":
        """A fresh state for a sub-agent, one level deeper.

        A new ledger, not a shared one: the sub-agent's gates, approvals and
        attempts are its own, and its usage is rolled up to the parent as a
        single number rather than by interleaving two runs into one log.
        """
        return RunState.begin(task, depth=self.depth + 1, parent_run_id=self.run_id)

    # -- the ledger, forwarded -------------------------------------------------
    # Middleware receive a RunState and never reach past it, so the two halves
    # cannot be passed around separately and get out of step.

    def append(self, seam: str, kind: str, key: str = "", **data) -> Event:
        return self.ledger.append(seam, kind, key, **data)

    def count(self, kind: str, key: str | None = None) -> int:
        return self.ledger.count(kind, key)

    def has(self, kind: str, key: str | None = None) -> bool:
        return self.ledger.has(kind, key)

    def last(self, kind: str, key: str | None = None) -> Event | None:
        return self.ledger.last(kind, key)

    def all(self, kind: str | None = None, key: str | None = None) -> list[Event]:
        return self.ledger.all(kind, key)

    # -- derived ---------------------------------------------------------------

    @property
    def artifact_hash(self) -> str:
        """The current artifact, read off the log rather than stored beside it.

        One source of truth, and the earlier hashes stay available — which is
        what lets you answer "what did the reviewer actually approve?" after
        three more edits have landed.
        """
        event = self.last("artifact_hashed")
        return event.key if event else ""

    @property
    def attempts(self) -> int:
        return self.count("model_call")

    @property
    def total_tokens(self) -> int:
        """Every token this run spent, including its sub-agents'.

        Sub-agents stamp their rollup here on return, so a budget read from
        this number is not quietly blind to the work it delegated.
        """
        return self.ledger.sum("model_call", "tokens") + self.ledger.sum(
            "subagent_returned", "tokens"
        )

    def stamp_artifact(self, digest: str) -> Event:
        """Record what the artifact looks like now.

        Called after anything that could have changed it. Gate results are bound
        to the hash they were checked against, so a new hash silently strands
        every approval earned against the old one — the agent cannot edit its
        way past a review it already passed.
        """
        return self.append("artifact", "artifact_hashed", key=digest)

    def summary(self) -> str:
        kinds: dict[str, int] = {}
        for event in self.ledger:
            kinds[event.kind] = kinds.get(event.kind, 0) + 1
        cells = " ".join(f"{k}={v}" for k, v in kinds.items())
        return (
            f"[run {self.run_id} d{self.depth} @{self.artifact_hash[:7] or '-------'}] "
            f"turns={self.attempts} tokens={self.total_tokens} {cells}"
        )

    # -- persistence -----------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "task": self.task,
            "depth": self.depth,
            "parent_run_id": self.parent_run_id,
            "workspace_ref": self.workspace_ref,
            "history_ref": self.history_ref,
            "ledger": self.ledger.to_dict(),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, raw: dict) -> "RunState":
        data = dict(raw)
        return cls(ledger=Ledger.from_dict(data.pop("ledger")), **data)

    @classmethod
    def from_json(cls, raw: str) -> "RunState":
        return cls.from_dict(json.loads(raw))
