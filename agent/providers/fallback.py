"""The ordered list of models to try, and which of them are resting.

One choice is one provider-and-model pair plus the settings for calling it. The
list is flat and read top to bottom: the first entry is the model you want, the
rest are what you settle for when it will not answer.

    [{"provider": "openai", "model": "gpt-5"},
     {"provider": "gemini", "model": "gemini-2.5-flash-lite", "timeout": 30},
     {"provider": "openai", "model": "qwen-3.5", "token_budget": 16_000}]
"""

from __future__ import annotations

import json
import os
import time

from pydantic import BaseModel, ConfigDict, Field

from .base import DEFAULT_TIMEOUT
from .model_registry import registry


DEFAULT_PROVIDER = "openai"

# Two attempts, not the four a single model used to get. The chain is the new
# way out of a bad model, and four attempts each across four models is sixteen
# calls and minutes of sleeping before anyone hears about it.
DEFAULT_MAX_ATTEMPTS = 2

# A cost ceiling, not the model's context limit — budgeting to the window would
# mean never compacting until a question had already cost a fortune.
DEFAULT_TOKEN_BUDGET = 60_000

# How long a model is left alone after failing. One that turned the call away
# needs real time to recover; one that merely ran out of retries is usually
# back sooner.
UNAVAILABLE_REST = 60.0
EXHAUSTED_REST = 15.0


class ProviderUnavailable(Exception):
    """A provider that could not even be built — no SDK, no credentials.

    Named so that classify_error reads it as unavailable and the chain moves
    on, rather than treating a laptop without a Gemini key as fatal.
    """


class AllModelsFailed(Exception):
    """Every model in the chain was tried and none of them answered.

    Carries the whole trail. Without it the caller sees only the last error,
    which is usually the least interesting one — a refused connection to a
    local server, when the story started with a rate limit two models earlier.
    """

    def __init__(self, failures: list[tuple[ModelChoice, Exception]]) -> None:
        self.failures = failures
        trail = "; ".join(
            f"{choice}: {type(error).__name__}: {error}" for choice, error in failures
        )
        super().__init__(f"all {len(failures)} model calls failed — {trail}")


class ModelChoice(BaseModel):
    """One provider-and-model pair, with the settings for calling it."""

    model_config = ConfigDict(frozen=True)

    provider: str
    # Left as None to take the provider's own default, so a chain can name a
    # provider without anyone having to know its model names.
    model: str | None = None
    # At least one, or this entry would be skipped without ever being called —
    # a way of disabling a model that reads like a typo.
    max_attempts: int = Field(default=DEFAULT_MAX_ATTEMPTS, ge=1)
    # A provider that hangs rather than fails never reaches the next choice at
    # all, so the fallback chain is only as good as this limit. None means no
    # limit, which is the SDKs' reading of it and rarely what anyone wants.
    timeout: float | None = Field(default=DEFAULT_TIMEOUT, gt=0)
    # Context windows differ down the chain, so the compaction threshold has to
    # as well: one fixed ceiling is either wasteful on the big model or too
    # generous for the small one.
    token_budget: int = Field(default=DEFAULT_TOKEN_BUDGET, gt=0)

    def model_post_init(self, context) -> None:
        # Reject an unknown provider now, while the traceback still points at
        # the configuration, rather than on the call that needed the fallback.
        registry.get(self.provider)

    @property
    def key(self) -> tuple[str, str | None]:
        """What a cooldown is remembered against."""
        return self.provider, self.model

    def __str__(self) -> str:
        return f"{self.provider}/{self.model or 'default'}"


def load_chain(chain: list[ModelChoice | dict] | None = None) -> list[ModelChoice]:
    """Build the list of models to try, in order.

    Three sources, first one that answers: what the caller passed, LLM_CHAIN as
    JSON in the environment, and finally the single LLM_PROVIDER/LLM_MODEL pair
    that was the only option before any of this existed. Reading it from the
    environment is what lets the eval scripts be pointed elsewhere untouched.
    """
    if chain is None:
        raw = os.getenv("LLM_CHAIN")
        chain = json.loads(raw) if raw else None
    if chain is None:
        chain = [
            {
                "provider": os.getenv("LLM_PROVIDER", DEFAULT_PROVIDER),
                "model": os.getenv("LLM_MODEL") or None,
            }
        ]
    if not chain:
        raise ValueError("A model chain needs at least one entry.")
    return [
        choice if isinstance(choice, ModelChoice) else ModelChoice(**choice)
        for choice in chain
    ]


class CooldownRecord:
    """Which models are resting, and until when.

    One instance is shared by every LLMService in the process. They all draw on
    the same accounts, so a rate limit one of them ran into is news to all of
    them — and paying the timeout again on every call to learn it is exactly
    what this saves. A race between threads costs one wasted call, so the plain
    dict is enough.
    """

    until: dict[tuple[str, str | None], float]

    def __init__(self) -> None:
        self.until = {}

    def rest(self, choice: ModelChoice, seconds: float) -> None:
        self.until[choice.key] = time.monotonic() + seconds

    def get_ready_models(self, chain: list[ModelChoice]) -> list[ModelChoice]:
        """The choices worth trying, best first.

        Resting models are skipped, but if that leaves nothing the whole chain
        comes back: skipping is meant to save a doomed call, not to refuse to
        make one.
        """
        now = time.monotonic()
        ready = [choice for choice in chain if self.until.get(choice.key, 0.0) <= now]
        return ready or chain


cooldowns = CooldownRecord()
