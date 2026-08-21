"""Whether anyone hears about a run.

This is the whole difference between the two patterns: a digest is delivered
because it ran, a monitor because something changed. Same trigger, same session,
same channel — only the policy differs, so neither pattern needs its own plumbing.
"""

from datetime import datetime, timedelta
from typing import Literal, Protocol

from pydantic import BaseModel

from .outcome import Outcome, StateRecord


class Decision(BaseModel):
    """A message to send. Its absence — None — is the interesting case."""

    kind: Literal["digest", "new", "reminder", "recovery"]
    subject: str
    body: str


class Policy(Protocol):
    def decide(
        self, outcome: Outcome, prior: StateRecord | None, now: datetime
    ) -> Decision | None:
        """The message this outcome warrants, or None to stay silent."""


class DeliverAlways(BaseModel):
    """Digest: the run's output is the point, so it always goes out."""

    def decide(
        self, outcome: Outcome, prior: StateRecord | None, now: datetime
    ) -> Decision | None:
        kind = "new" if outcome.status == "error" else "digest"
        return Decision(
            kind=kind, subject=outcome.title or "Digest", body=outcome.body
        )


class DeliverOnTransition(BaseModel):
    """Monitoring: alert on the edge, not on the level.

    Still-broken is not news; broken-since-last-run is. Without this, a check
    every 15 minutes sends 96 identical messages a day and gets muted — which is
    the failure mode that makes people stop trusting monitors altogether.
    """

    # None to never repeat. Some reminder is usually right: a problem that broke
    # at 2am and alerted once is invisible by the time anyone is awake.
    renotify_after: timedelta | None = timedelta(hours=6)
    notify_recovery: bool = True

    def decide(
        self, outcome: Outcome, prior: StateRecord | None, now: datetime
    ) -> Decision | None:
        if outcome.status == "ok":
            if prior and prior.status != "ok" and self.notify_recovery:
                return Decision(
                    kind="recovery",
                    subject=f"Recovered: {prior.title}",
                    body=outcome.body,
                )
            return None

        # "error" lands here with "alert" on purpose: a monitor that cannot run
        # is a fact about the system too, and silence would hide it.
        if prior is None or prior.status == "ok":
            return Decision(kind="new", subject=outcome.title, body=outcome.body)

        if (
            self.renotify_after is not None
            and prior.last_delivered is not None
            and now - prior.last_delivered >= self.renotify_after
        ):
            return Decision(
                kind="reminder",
                subject=f"Still: {outcome.title}",
                body=outcome.body,
            )

        # Nothing changed and nothing is due to repeat. This branch is the
        # feature — it is what "silent success, loud failure" actually costs.
        return None
