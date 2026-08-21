"""Execute one fire of one task: claim, run isolated, decide, deliver, record."""

import logging
import threading
from datetime import datetime, timezone
from typing import Callable

from agent.conversation import Conversation

from .outcome import Outcome, OutcomeSink
from .store import SqliteStore
from .task import ScheduledTask

logger = logging.getLogger(__name__)

# Every outcome needs a state key, including the healthy ones, or an "ok" run
# would have nothing to compare the next alert against.
DEFAULT_FINGERPRINT = "-"


def run_once(
    task: ScheduledTask, store: SqliteStore, scheduled_for: datetime
) -> Outcome | None:
    """Run one fire end to end. None means another worker already had it."""
    if not store.claim(task.id, scheduled_for):
        logger.info(
            "Task %s at %s is already claimed", task.id, scheduled_for.isoformat()
        )
        return None

    outcome = _execute(task)
    now = datetime.now(timezone.utc)
    fingerprint = outcome.fingerprint or DEFAULT_FINGERPRINT
    prior = store.get_state(task.id, fingerprint)

    delivered = False
    if decision := task.policy.decide(outcome, prior, now):
        try:
            task.channel.send(decision.subject, decision.body)
            delivered = True
            logger.info("Task %s delivered (%s)", task.id, decision.kind)
        except Exception:
            # Deliberately not recorded as delivered: the next run should treat
            # the alert as new and try again, rather than suppress it as a
            # duplicate of a message that never arrived.
            logger.exception(
                "Task %s: delivery via %s failed",
                task.id,
                type(task.channel).__name__,
            )
    else:
        logger.info("Task %s: %s, nothing to deliver", task.id, outcome.status)

    store.put_state(task.id, fingerprint, outcome, now, delivered=delivered)
    store.finish(task.id, scheduled_for, outcome.status)
    return outcome


def _execute(task: ScheduledTask) -> Outcome:
    """Build a fresh session, run it once, and return what it reported."""
    sink = OutcomeSink()
    try:
        agent = task.build_agent(sink)
        # A conversation with no history, used for exactly one question and then
        # dropped. That is the whole of session isolation here — Agent holds
        # nothing per-session, so there is nothing else to reset.
        conversation = Conversation(agent, token_budget=task.token_budget)
        _with_deadline(
            lambda: conversation.ask(task.prompt), task.deadline_seconds
        )
    except Exception as error:
        # A monitor that dies quietly is worse than no monitor, so the failure
        # takes the same path a breach would rather than ending in the log.
        logger.exception("Task %s failed", task.id)
        return Outcome(
            status="error",
            title=f"{task.id} failed",
            body=f"{type(error).__name__}: {error}",
            fingerprint=f"run-failure:{task.id}",
        )

    if sink.outcome is None:
        # It answered, but never called report. Treated as a failure because the
        # alternative is a monitor that has been silently useless for weeks.
        return Outcome(
            status="error",
            title=f"{task.id} reported nothing",
            body="The run finished without calling the report tool.",
            fingerprint=f"no-report:{task.id}",
        )

    return sink.outcome


def _with_deadline(work: Callable[[], object], seconds: int) -> None:
    """Run work, giving up on it after `seconds`.

    A soft deadline: Python cannot kill a running thread, so an abandoned run
    keeps going until its own I/O returns. What it does buy is that the schedule
    moves on — one hung run cannot occupy a worker forever. The thread is a
    daemon so it can never hold up shutdown. Run tasks in separate processes if
    you need the stronger guarantee.
    """
    box: dict[str, object] = {}

    def target() -> None:
        try:
            box["value"] = work()
        except BaseException as error:  # re-raised on the caller's thread below
            box["error"] = error

    thread = threading.Thread(target=target, daemon=True, name="scheduled-run")
    thread.start()
    thread.join(seconds)

    if thread.is_alive():
        raise TimeoutError(f"run exceeded its {seconds}s deadline")
    if "error" in box:
        raise box["error"]
