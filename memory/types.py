"""Vocabulary of the memory store: four types, three verbs, one confidence rubric.

Every constant here exists because something branches on it.

Four types, not twelve. Finer-grained labels (fact, preference, goal, habit,
constraint…) overlap at the boundaries, so the same sentence gets labelled
differently on two runs and every retrieval filter built on the label goes
unreliable. What actually differs is four things, so there are four types.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


class MemoryType(StrEnum):
    """What a memory is, chosen so that behaviour follows from the label.

    DIRECTIVE   how to respond           preference, constraint, instruction
    PROFILE     who the user is          fact, skill, relationship, interest
    CURRENT     what they are doing now  goal, project, situational context
    EVENT       what happened            a dated, historical occurrence
    """

    DIRECTIVE = "directive"
    PROFILE = "profile"
    CURRENT = "current"
    EVENT = "event"


class Operation(StrEnum):
    """What one row does to the log. Every one of them is an INSERT."""

    ADD = "add"
    UPDATE = "update"
    INVALIDATE = "invalidate"


# Stands in for the user system that does not exist yet. Every query is already
# scoped by it, so the seam is real even while the value is fixed.
DEFAULT_USER_ID = "local-user"

# Only directives are worth prompt tokens on every turn. The rest are retrieved
# when a question calls for them.
INJECTED_TYPES = frozenset({MemoryType.DIRECTIVE})


# -- confidence -------------------------------------------------------------
# An ordinal, not a float. Asked for 0-1, models return 0.7-0.9 for nearly
# everything; asked to place a statement on three anchored rungs they are far
# more consistent. The anchor is how directly the user said it, which is
# observable in the transcript rather than a judgment about the world.

STATED = 3  # asserted outright, in their own words
IMPLIED = 2  # entailed by their words, but not stated as fact
INFERRED = 1  # read off behaviour or one observation; easily wrong

# A guess should not shape every reply, so the always-injected block starts at
# IMPLIED. INFERRED memories stay searchable, they just do not assert themselves.
MIN_INJECT_CONFIDENCE = IMPLIED


# -- expiry -----------------------------------------------------------------
# Derived at insert from the type, never set by a model: a model-supplied TTL is
# a plausible-looking number with nothing behind it.

TTL_DAYS: dict[MemoryType, int | None] = {
    MemoryType.DIRECTIVE: None,  # holds until the user says otherwise
    MemoryType.PROFILE: None,  # superseded, not expired
    MemoryType.CURRENT: 90,  # what someone is working on stops being true
    MemoryType.EVENT: None,  # historical, so it stays true forever
}

# A guess nothing ever reconfirms decays out instead of asserting itself forever.
INFERRED_TTL_DAYS = 30


def expires_at(
    memory_type: MemoryType, confidence: int, now: datetime | None = None
) -> datetime | None:
    """When this memory stops counting, or None if it does not."""
    now = now or datetime.now(timezone.utc)
    days = TTL_DAYS[MemoryType(memory_type)]
    if confidence <= INFERRED:
        days = INFERRED_TTL_DAYS if days is None else min(days, INFERRED_TTL_DAYS)
    return now + timedelta(days=days) if days else None


# -- what the writer proposes, and what it decides to do with it -------------


class MemoryContent(BaseModel):
    """One candidate memory, before anything has been looked up.

    Not a row: it carries no id and no verb, because whether it becomes an
    insert, a supersede or nothing is not the proposer's decision.
    """

    content: str = Field(min_length=1)
    memory_type: MemoryType
    confidence: int = Field(default=IMPLIED, ge=INFERRED, le=STATED)


class MemoryOperation(BaseModel):
    """What the writer decided to do with one MemoryContent, from four verbs.

    Not the same thing as `Operation` above: that is what a row *is* on the log
    (add / update / invalidate, every one an INSERT). This is a decision, and it
    has a fourth verb the log has no row for.

    `noop` is a first-class outcome, not an absence: a model handed three write
    verbs will find a way to use one. Store size is the whole game — what ruins
    a memory system is four hundred junk rows drowning retrieval, not one missed
    fact.
    """

    verb: Literal["add", "update", "invalidate", "noop"]
    # Which existing memory the verb acts on. Required for update and
    # invalidate, and always checked against the candidates actually shown.
    target_id: int | None = None
    reason: str = ""


class MemoryContents(BaseModel):
    """What one extraction returns.

    A wrapper because structured output needs an object at the top level, and
    because naming the field is what lets the prompt say "return a list of
    memories" and have the schema agree.
    """

    memories: list[MemoryContent] = Field(default_factory=list)
