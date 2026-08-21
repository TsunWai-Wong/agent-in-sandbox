"""The two facts that have to outlive the process: which fires already ran, and
what the last run said.

SQLite because it is the smallest thing with a real transaction, and `claim`
below is a transaction rather than a check — a read-then-write would let two
workers both decide the fire was theirs. Everything the runner needs is the five
methods here, so moving to Postgres to run workers on more than one box is a new
class implementing them, not a new design.
"""

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .outcome import Outcome, StateRecord

DEFAULT_LEASE_SECONDS = 900

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    task_id       TEXT NOT NULL,
    scheduled_for TEXT NOT NULL,
    status        TEXT NOT NULL,
    started_at    TEXT NOT NULL,
    finished_at   TEXT,
    PRIMARY KEY (task_id, scheduled_for)
);

CREATE TABLE IF NOT EXISTS task_state (
    task_id        TEXT NOT NULL,
    fingerprint    TEXT NOT NULL,
    status         TEXT NOT NULL,
    title          TEXT NOT NULL DEFAULT '',
    last_seen      TEXT NOT NULL,
    last_delivered TEXT,
    PRIMARY KEY (task_id, fingerprint)
);
"""


def _iso(when: datetime) -> str:
    """UTC ISO-8601, which sorts lexicographically — so SQL can compare times."""
    return when.astimezone(timezone.utc).isoformat()


class SqliteStore:
    """Run bookkeeping and per-fingerprint state, in one file."""

    def __init__(self, path: str | Path = "scheduler.db") -> None:
        self.path = str(path)
        with self._connect() as conn:
            # WAL so the clock reading and a worker writing do not block each
            # other; it is also what makes several worker threads worth having.
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(SCHEMA)

    @contextmanager
    def _connect(self):
        # A connection per call. They are cheap, and it sidesteps sqlite3's
        # one-thread-per-connection rule without a lock of our own.
        conn = sqlite3.connect(self.path, timeout=30)
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def claim(
        self,
        task_id: str,
        scheduled_for: datetime,
        lease_seconds: int = DEFAULT_LEASE_SECONDS,
    ) -> bool:
        """Take ownership of one (task, fire) pair. False means someone else has it.

        The key is the SCHEDULED instant, not the current one. A retry, a second
        clock process and a restart all recompute the same key, which is what
        turns at-least-once execution into exactly-once delivery.

        The lease covers the other half: a worker killed mid-run leaves its row
        marked running forever, so a claim older than the lease can be taken over.
        Set it longer than the task's deadline, or two workers will overlap.
        """
        now = datetime.now(timezone.utc)
        expired_before = _iso(now - timedelta(seconds=lease_seconds))
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO runs (task_id, scheduled_for, status, started_at)
                VALUES (?, ?, 'running', ?)
                ON CONFLICT (task_id, scheduled_for) DO UPDATE SET
                    started_at = excluded.started_at,
                    status = 'running'
                WHERE runs.status = 'running' AND runs.started_at < ?
                """,
                (task_id, _iso(scheduled_for), _iso(now), expired_before),
            )
            return cursor.rowcount == 1

    def finish(self, task_id: str, scheduled_for: datetime, status: str) -> None:
        """Close out a claimed run."""
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE runs SET status = ?, finished_at = ?
                WHERE task_id = ? AND scheduled_for = ?
                """,
                (status, _iso(datetime.now(timezone.utc)), task_id, _iso(scheduled_for)),
            )

    def get_state(self, task_id: str, fingerprint: str) -> StateRecord | None:
        """What the last run said about this fingerprint, if anything."""
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT status, title, last_seen, last_delivered FROM task_state
                WHERE task_id = ? AND fingerprint = ?
                """,
                (task_id, fingerprint),
            ).fetchone()

        if row is None:
            return None
        status, title, last_seen, last_delivered = row
        return StateRecord(
            status=status,
            title=title,
            last_seen=datetime.fromisoformat(last_seen),
            last_delivered=(
                datetime.fromisoformat(last_delivered) if last_delivered else None
            ),
        )

    def put_state(
        self,
        task_id: str,
        fingerprint: str,
        outcome: Outcome,
        now: datetime,
        delivered: bool,
    ) -> None:
        """Record this run as the new prior for its fingerprint.

        last_delivered is sticky: it is only advanced when something actually
        went out, because it answers "when were they last told", and a
        suppressed run told nobody anything.
        """
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO task_state
                    (task_id, fingerprint, status, title, last_seen, last_delivered)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT (task_id, fingerprint) DO UPDATE SET
                    status = excluded.status,
                    title = excluded.title,
                    last_seen = excluded.last_seen,
                    last_delivered = COALESCE(
                        excluded.last_delivered, task_state.last_delivered
                    )
                """,
                (
                    task_id,
                    fingerprint,
                    outcome.status,
                    outcome.title,
                    _iso(now),
                    _iso(now) if delivered else None,
                ),
            )
