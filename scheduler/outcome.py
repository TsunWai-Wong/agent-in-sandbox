"""What one scheduled run decided, and how the agent says so.

Free text is enough for a digest and useless for a monitor: "tell me only when
something changed" is a comparison between runs, and prose cannot be compared.
So a run ends in a typed Outcome, and the agent produces one by calling a tool —
the registry already turns a handler's signature into a schema, so the model is
told the exact shape to produce instead of being asked for JSON and trusted.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict

Status = Literal["ok", "alert", "error"]

REPORT_INSTRUCTION = """
Finish by calling the report tool exactly once, then reply "done" without
calling any more tools. A run that ends without calling report is a failed run,
however good its prose.
""".strip()


class Outcome(BaseModel):
    """One run's verdict."""

    model_config = ConfigDict(frozen=True)

    status: Status
    title: str = ""
    body: str = ""
    # What the alert is ABOUT, not that it happened: "disk:/var" stays identical
    # across every hour that disk stays full, which is exactly what lets the
    # policy tell an ongoing problem from a new one. Put prose in here — a
    # percentage, a timestamp — and every run looks like a fresh problem.
    fingerprint: str = ""


class StateRecord(BaseModel):
    """What the previous run said about this fingerprint, as the store kept it."""

    model_config = ConfigDict(frozen=True)

    status: Status
    title: str = ""
    last_seen: datetime
    # None when the outcome was never delivered — suppressed as a duplicate, or
    # the channel failed. Either way there is no "already told them" to lean on.
    last_delivered: datetime | None = None


class OutcomeSink:
    """Collects the verdict of one run.

    One instance per run and never shared: it is mutable, and two runs writing
    into the same sink is one run alerting with the other's numbers.
    """

    outcome: Outcome | None

    def __init__(self) -> None:
        self.outcome = None

    def report(
        self, status: str, title: str, body: str, fingerprint: str = ""
    ) -> str:
        """Report the result of this check. Call exactly once, at the end.

        Args:
            status: "ok" if everything you checked is within its threshold,
                "alert" if any of it is breached.
            title: One line naming what you found; used as the alert subject.
            body: The detail behind the verdict, including the numbers you read.
            fingerprint: Stable identifier for WHAT is wrong, such as
                "disk:/var". Use the same string on every run that finds the
                same problem, and leave it empty when the status is ok.
        """
        if status not in ("ok", "alert"):
            # "error" is the runner's to report and never the model's: it means
            # the run did not complete, and a model that answered cannot know
            # that about itself.
            return "Error: status must be 'ok' or 'alert'."

        self.outcome = Outcome(
            status=status, title=title, body=body, fingerprint=fingerprint
        )
        return "Reported. You are done — reply 'done' without calling more tools."
