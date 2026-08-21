"""When a scheduled task fires.

Every trigger answers one question — given an instant, when is the next one? —
so the clock loop never branches on trigger type. None means "never again",
which is how a one-shot retires itself instead of needing a special case.
"""

from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import Protocol
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field, field_validator

# minute, hour, day-of-month, month, day-of-week
FIELD_BOUNDS = ((0, 59), (0, 23), (1, 31), (1, 12), (0, 6))

# A cron expression is matched by walking wall-clock minutes forward, which is
# both simple and correct. The cap is what stops an expression that can never
# match (31 February) from walking forever.
SEARCH_LIMIT_MINUTES = 366 * 24 * 60


class Trigger(Protocol):
    """A schedule, reduced to the only question the clock asks of it."""

    def next_after(self, t: datetime) -> datetime | None:
        """The first firing instant strictly after t, in UTC, or None if never."""


def _parse_field(spec: str, low: int, high: int) -> frozenset[int]:
    """Expand one cron field — `*`, `5`, `1-4`, `*/15`, `0,30` — into its values."""
    values: set[int] = set()

    for part in spec.split(","):
        step = 1
        if "/" in part:
            part, _, step_text = part.partition("/")
            step = int(step_text)
            if step < 1:
                raise ValueError(f"step must be positive in cron field {spec!r}")
        if part == "*":
            start, end = low, high
        elif "-" in part:
            start_text, _, end_text = part.partition("-")
            start, end = int(start_text), int(end_text)
        else:
            start = end = int(part)
        values.update(range(start, end + 1, step))

    if high == 6:
        # 7 is a second spelling of Sunday in every cron implementation. Folding
        # it here means the matcher below never has to know that.
        values = {value % 7 for value in values}

    if out_of_range := sorted(v for v in values if not low <= v <= high):
        raise ValueError(
            f"cron field {spec!r} out of range {low}-{high}: {out_of_range}"
        )
    return frozenset(values)


@lru_cache(maxsize=128)
def _parse_expr(expr: str) -> tuple[frozenset[int], ...]:
    """Parse a five-field cron expression. Cached: a task re-parses it every fire."""
    parts = expr.split()
    if len(parts) != 5:
        raise ValueError(
            f"cron expression needs 5 fields, got {len(parts)}: {expr!r}"
        )
    return tuple(
        _parse_field(part, low, high)
        for part, (low, high) in zip(parts, FIELD_BOUNDS)
    )


def _matches(when: datetime, fields: tuple[frozenset[int], ...]) -> bool:
    """Whether a wall-clock minute satisfies a parsed expression."""
    minutes, hours, days, months, weekdays = fields

    if when.minute not in minutes or when.hour not in hours:
        return False
    if when.month not in months:
        return False

    # Python counts Monday as 0, cron counts Sunday as 0.
    on_day = when.day in days
    on_weekday = (when.weekday() + 1) % 7 in weekdays

    # Cron's one irregularity: when both day fields are restricted they are ORed
    # rather than ANDed, so "0 0 1 * 1" means the 1st of the month OR any Monday.
    if len(days) < 31 and len(weekdays) < 7:
        return on_day or on_weekday
    return on_day and on_weekday


class CronTrigger(BaseModel):
    """Standard five-field cron, evaluated in the task's own timezone.

    The timezone is a field rather than the host's clock for two reasons: "8:00
    AM" is meaningless without one, and a cron evaluated in UTC drifts by an
    hour twice a year — quietly turning the 8am digest into a 9am one.
    """

    expr: str
    tz: str = "UTC"

    @field_validator("expr")
    @classmethod
    def _must_parse(cls, value: str) -> str:
        # At construction, so a typo fails where the task is declared rather
        # than in a worker thread at 3am.
        _parse_expr(value)
        return value

    @field_validator("tz")
    @classmethod
    def _must_be_a_zone(cls, value: str) -> str:
        ZoneInfo(value)
        return value

    def next_after(self, t: datetime) -> datetime:
        zone = ZoneInfo(self.tz)
        fields = _parse_expr(self.expr)

        # Stepped as a naive wall clock and localized only at the end. Cron means
        # wall-clock time, and adding a timedelta to an aware datetime is
        # absolute-time arithmetic: across a DST boundary it would move the clock
        # face by two hours, not one, and the schedule would skip a day.
        local = t.astimezone(zone).replace(tzinfo=None, second=0, microsecond=0)

        for _ in range(SEARCH_LIMIT_MINUTES):
            local += timedelta(minutes=1)
            if not _matches(local, fields):
                continue
            fire = local.replace(tzinfo=zone).astimezone(timezone.utc)
            # A wall-clock minute can repeat (DST fall back) or not exist at all
            # (spring forward). Demanding strict progress is what keeps the
            # repeated hour from firing the same schedule twice.
            if fire > t:
                return fire

        raise ValueError(f"cron expression never fires within a year: {self.expr!r}")


class IntervalTrigger(BaseModel):
    """Every N seconds from whenever the last fire was scheduled."""

    seconds: int = Field(gt=0)

    def next_after(self, t: datetime) -> datetime:
        return t + timedelta(seconds=self.seconds)


class OnceTrigger(BaseModel):
    """A single fire at a fixed instant — "remind me in 20 minutes"."""

    at: datetime

    @field_validator("at")
    @classmethod
    def _must_be_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("OnceTrigger.at needs a timezone")
        return value

    def next_after(self, t: datetime) -> datetime | None:
        return self.at if self.at > t else None
