import logging
import os
import random
import time

from dotenv import load_dotenv
from pydantic import BaseModel

from agent.tool_registry import ToolSchema
from monitoring import get_tracer

from .providers import (
    Attachment,
    ChatResponse,
    Message,
    Provider,
    registry,
)
from .providers.errors import ErrorClass, classify_error
from .providers.fallback import (
    DEFAULT_PROVIDER,
    EXHAUSTED_REST,
    UNAVAILABLE_REST,
    AllModelsFailed,
    CooldownRecord,
    ModelChoice,
    ProviderUnavailable,
    load_chain,
)
from .providers.fallback import cooldowns as shared_cooldowns


logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)

BACKOFF_SECONDS = 2.0


class LLMService:
    """Provider-agnostic entry point for model calls.

    Owns the cross-cutting concerns — retries, fallback, backoff, logging — so
    that every adapter under providers/ is left with request and response
    translation and nothing else. Which model answers is decided here and
    nowhere else.

    It works down a chain of models rather than holding one. When the preferred
    model will not answer, the next one takes the call; when it is a model that
    turned the call away, it is left resting for a while so the following calls
    do not pay its timeout again to learn the same thing. Every call still
    starts from the top of the chain, so a run that had to settle for a backup
    is back on its first choice the moment that one recovers.
    """

    chain: list[ModelChoice]
    cooldowns: CooldownRecord
    backoff_seconds: float

    def __init__(
        self,
        chain: list[ModelChoice | dict] | None = None,
        model: str | None = None,
        backoff_seconds: float = BACKOFF_SECONDS,
        cooldowns: CooldownRecord | None = None,
    ) -> None:
        """Build a service around a chain of models to try in order.

        Passing `model` alone pins the service to that one model with no
        fallback, which is what a caller who has already chosen — a reviewer, a
        summarizer — is asking for. Passing neither reads the chain from the
        environment, so the eval scripts can be pointed elsewhere untouched.
        """
        load_dotenv()
        if chain is None and model is not None:
            chain = [
                {
                    "provider": os.getenv("LLM_PROVIDER", DEFAULT_PROVIDER),
                    "model": model,
                }
            ]
        self.chain = load_chain(chain)
        # Shared by default: every LLMService in the process draws on the same
        # accounts, so one of them discovering a rate limit should spare the
        # rest from rediscovering it.
        self.cooldowns = cooldowns or shared_cooldowns
        self.backoff_seconds = backoff_seconds
        self._providers: dict[tuple[str, str | None], Provider] = {}

    @property
    def token_budget(self) -> int:
        """The budget of the model most likely to take the next call.

        An estimate, not a promise: if the call ends up falling through to a
        model with a smaller window, that overflow reads as unavailable and the
        chain simply moves on again.
        """
        return self.cooldowns.get_ready_models(self.chain)[0].token_budget

    def chat(
        self,
        messages: list[Message],
        system: str | None = None,
        tools: list[ToolSchema] | None = None,
        text_format: type[BaseModel] | None = None,
        model: str | None = None,
    ) -> ChatResponse:
        """Call the best available model, working down the chain as needed.

        Pass text_format (a Pydantic model) for structured output; the parsed
        object arrives on the returned response as response.parsed. Pass tools
        for agentic tool calling, and system for the instruction — never as an
        entry in messages, since providers place it differently. Passing model
        pins this one call to that model, with no fallback.

        The whole thing sits in one span, waiting included. Auto-instrumentation
        already times each attempt; what it cannot show is the sleeping between
        them, so a rate-limited call would otherwise read as an unexplained gap.

        Raises whatever the provider raised if it was a bad request, and
        AllModelsFailed once the chain runs out.
        """
        with tracer.start_as_current_span(
            "llm.chat", openinference_span_kind="chain"
        ) as span:
            chain = self._chain_for(model)
            failures: list[tuple[ModelChoice, Exception]] = []
            waited = 0.0

            for choice in self.cooldowns.get_ready_models(chain):
                for attempt in range(1, choice.max_attempts + 1):
                    try:
                        provider = self._build_provider(choice)
                        response = provider.chat(
                            messages=messages,
                            system=system,
                            tools=tools,
                            text_format=text_format,
                            model=choice.model,
                        )
                    except Exception as error:
                        failures.append((choice, error))
                        error_class = classify_error(error)

                        # A request every model would reject identically.
                        # Walking the chain would only bill for the same
                        # mistake several times over.
                        if error_class is ErrorClass.PERMANENT:
                            raise

                        if error_class is ErrorClass.UNAVAILABLE:
                            self._step_down(
                                span, choice, error, "unavailable", UNAVAILABLE_REST
                            )
                            break
                        if attempt == choice.max_attempts:
                            self._step_down(
                                span, choice, error, "out of attempts", EXHAUSTED_REST
                            )
                            break
                        waited += self._backoff(span, choice, error, attempt)
                    else:
                        # Depth counts from the full chain, not from what was
                        # left after resting models were skipped: a backup that
                        # answers because the first choice was skipped is still
                        # a backup, and must not read as depth 0.
                        self._record(
                            span,
                            choice,
                            provider,
                            chain.index(choice),
                            attempt - 1,
                            waited,
                        )
                        return response

            raise AllModelsFailed(failures) from failures[-1][1]

    def _chain_for(self, model: str | None) -> list[ModelChoice]:
        """The chain for one call. A model override pins it to that model."""
        if model is None:
            return self.chain
        return [self.chain[0].model_copy(update={"model": model})]

    def _build_provider(self, choice: ModelChoice) -> Provider:
        """Build the adapter for a choice the first time it is needed.

        Lazily, because constructing one opens a client and reads an API key:
        building the whole chain up front would mean nobody could start without
        credentials for every provider in it. A provider that cannot be built
        is reported as unavailable rather than fatal, so a missing SDK costs
        the chain one entry instead of the whole call.
        """
        if choice.key not in self._providers:
            try:
                self._providers[choice.key] = registry.get(choice.provider)(
                    model=choice.model, timeout=choice.timeout
                )
            except Exception as error:
                raise ProviderUnavailable(
                    f"{choice} could not be built: {error}"
                ) from error
        return self._providers[choice.key]

    def _step_down(
        self,
        span,
        choice: ModelChoice,
        error: Exception,
        reason: str,
        rest_seconds: float,
    ) -> None:
        """Rest this model and hand the call to the next one in the chain."""
        self.cooldowns.rest(choice, rest_seconds)
        span.add_event(
            "failover",
            {"model": str(choice), "reason": reason, "error": type(error).__name__},
        )
        logger.warning(
            "%s %s (%s); resting it %.0fs and trying the next model",
            choice,
            reason,
            type(error).__name__,
            rest_seconds,
        )

    def _backoff(
        self, span, choice: ModelChoice, error: Exception, attempt: int
    ) -> float:
        """Sleep before asking the same model again, and return what that cost."""
        delay = self.backoff_seconds * 2 ** (attempt - 1)
        # Equal jitter: half the wait fixed, half random. Sub-agents fan out
        # four at a time onto one provider, and a delay that is purely a
        # function of the attempt number brings all four back on the same
        # second to take the same rate limit again.
        delay = delay / 2 + random.uniform(0, delay / 2)
        span.add_event(
            "retry",
            {
                "model": str(choice),
                "attempt": attempt,
                "error": type(error).__name__,
                "delay_s": round(delay, 3),
            },
        )
        logger.warning(
            "%s failed (%s), attempt %d/%d, retrying in %.1fs",
            choice,
            type(error).__name__,
            attempt,
            choice.max_attempts,
            delay,
        )
        time.sleep(delay)
        return delay

    @staticmethod
    def _record(
        span,
        choice: ModelChoice,
        provider: Provider,
        depth: int,
        retries: int,
        waited: float,
    ) -> None:
        """Note which model answered, and what it took to get there.

        fallback_depth is 0 when the first choice answered. Without it a quiet
        slide onto the cheap backup model would never show up anywhere.
        """
        span.set_attribute("llm.provider", choice.provider)
        span.set_attribute("llm.model", choice.model or provider.default_model)
        span.set_attribute("llm.fallback_depth", depth)
        span.set_attribute("retry.count", retries)
        span.set_attribute("retry.wait_seconds", round(waited, 3))

    @staticmethod
    def user_message(text: str, files: list[str] | None = None) -> Message:
        """Build one user turn.

        No longer a forward to the provider: a Message is neutral, so this is
        construction rather than translation. Kept here because every caller
        already had it, and because attachments are named by reference — the
        adapter is what decides how to carry the bytes.
        """
        return Message(
            role="user",
            text=text,
            attachments=[Attachment(ref=file) for file in (files or [])],
        )
