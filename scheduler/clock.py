"""The loop that turns triggers into runs.

It does one thing — find the fires that are due and hand them to a worker —
because that is the seam that makes this scale. Replace the pool with a queue
and the workers move to another process without anything else here changing.
"""

import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone

from .runner import run_once
from .store import SqliteStore
from .task import ScheduledTask

logger = logging.getLogger(__name__)

TICK_SECONDS = 1.0


class Scheduler:
    """In-process clock and worker pool over a set of tasks."""

    def __init__(
        self,
        store: SqliteStore,
        tasks: tuple[ScheduledTask, ...] = (),
        max_workers: int = 4,
        tick_seconds: float = TICK_SECONDS,
    ) -> None:
        self.store = store
        self.tick_seconds = tick_seconds
        self.tasks: dict[str, ScheduledTask] = {}
        # Next fire per task, held in memory. Losing it on restart is safe
        # because the store owns the fact that matters — whether a fire already
        # ran — and this is only ever a plan for the future.
        self._next: dict[str, datetime | None] = {}
        self._pool = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="run"
        )
        self._stop = threading.Event()

        for task in tasks:
            self.add(task)

    def add(self, task: ScheduledTask, now: datetime | None = None) -> None:
        now = now or datetime.now(timezone.utc)
        self.tasks[task.id] = task
        self._next[task.id] = task.trigger.next_after(now)
        logger.info("Task %s scheduled for %s", task.id, self._next[task.id])

    def tick(self, now: datetime | None = None) -> int:
        """Dispatch every fire that is due. Returns how many went out."""
        now = now or datetime.now(timezone.utc)
        dispatched = 0

        for task_id, fire_at in list(self._next.items()):
            if fire_at is None or fire_at > now:
                continue
            task = self.tasks[task_id]
            # Advanced from now rather than from fire_at, so a process that was
            # down over lunch does not replay every fire it missed. A digest six
            # hours late is noise; the next one is the one that matters.
            self._next[task_id] = task.trigger.next_after(now)
            self._submit(task, fire_at)
            dispatched += 1

        return dispatched

    def run_forever(self) -> None:
        """Tick until stop() or Ctrl-C, then let running tasks finish."""
        logger.info("Scheduler started with %d task(s)", len(self.tasks))
        try:
            while not self._stop.wait(self.tick_seconds):
                self.tick()
        except KeyboardInterrupt:
            logger.info("Interrupted; waiting for running tasks")
        finally:
            self._pool.shutdown(wait=True)

    def stop(self) -> None:
        self._stop.set()

    def _submit(self, task: ScheduledTask, fire_at: datetime) -> None:
        future = self._pool.submit(run_once, task, self.store, fire_at)
        # run_once turns task failures into outcomes, so anything landing here is
        # the scheduler's own bug — and a Future swallows it in silence.
        future.add_done_callback(lambda done: self._log_failure(task.id, done))

    @staticmethod
    def _log_failure(task_id: str, done: Future) -> None:
        if error := done.exception():
            logger.error("Dispatching task %s failed: %r", task_id, error)
