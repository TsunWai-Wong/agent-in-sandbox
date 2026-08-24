"""A session: one conversation, its user, and the log that is all of its state.

What Conversation used to be, minus the two things that were never its business.
History is gone — it is events now. The compaction policy is gone — it moved to
a middleware at before_model, where it fires mid-run instead of only after an
answer, and applies to callers that never built a Conversation at all.

Nothing here is a counter. Usage and last-activity are read off the log, for the
same reason RunState derives attempts: a number stored beside the log it
summarizes is a second answer waiting to disagree.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

from .event_store import EventStore, RunState


ACTIVE = "active"
CLOSED = "closed"


class SessionClosed(Exception):
    """A run was started on a session that has ended."""


@dataclass
class Session:
    """One conversation. The single place its state lives.

    `status` is the only stored field that is not identity, and it earns it by
    being branched on: a closed session refuses new runs.

    `user_id` may be None, and that is meaningful rather than missing — nobody
    owns this conversation, so there is no one to attribute a memory to. That is
    what a sub-agent, an eval script or a one-shot ask() gets, and it is why the
    default is None rather than a fallback id: a hundred eval runs must not write
    themselves into a real person's profile.
    """

    id: str
    user_id: str | None = None
    event_store: EventStore = field(default_factory=EventStore)
    created_at: float = field(default_factory=time.time)
    status: str = ACTIVE

    @classmethod
    def begin(cls, user_id: str | None = None) -> "Session":
        return cls(id=uuid.uuid4().hex, user_id=user_id)

    def begin_run(self, task: str) -> RunState:
        if self.status == CLOSED:
            raise SessionClosed(f"session {self.id} is closed")
        return RunState.begin(task, event_store=self.event_store)

    def close(self) -> None:
        """Idempotent, and does not itself write memory — only the caller knows
        whether it wants to wait for that."""
        self.status = CLOSED

    @property
    def updated_at(self) -> float:
        """Derived here; denormalize onto the session row in a durable store,
        where "list my conversations, newest first" would otherwise scan events."""
        events = list(self.event_store)
        return events[-1].at if events else self.created_at

    def usage(self) -> dict[str, int]:
        """What this conversation has cost, every run and sub-agent.

        No run_id filter, which is the difference from RunState.total_tokens —
        and why subagent_returned is excluded here. Sub-agent model calls are
        already in this log under their own run_id; counting the rollup too
        would bill them twice.
        """
        return {
            "input_tokens": self.event_store.sum("model_call", "input_tokens"),
            "output_tokens": self.event_store.sum("model_call", "output_tokens"),
            "total_tokens": self.event_store.sum("model_call", "tokens"),
        }

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "created_at": self.created_at,
            "status": self.status,
            "event_store": self.event_store.to_dict(),
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "Session":
        data = dict(raw)
        return cls(
            event_store=EventStore.from_dict(data.pop("event_store")), **data
        )
