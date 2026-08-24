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


logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)

DEFAULT_PROVIDER = "openai"
MAX_ATTEMPTS = 4
BACKOFF_SECONDS = 2.0


class LLMService:
    """Provider-agnostic entry point for model calls.

    Owns the cross-cutting concerns — retries, backoff, logging — so that every
    adapter under providers/ is left with request and response translation and
    nothing else. Which provider answers is decided here and nowhere else.
    """

    provider: Provider
    model: str | None
    max_attempts: int
    backoff_seconds: float

    def __init__(
        self,
        provider: Provider | str | None = None,
        model: str | None = None,
        max_attempts: int = MAX_ATTEMPTS,
        backoff_seconds: float = BACKOFF_SECONDS,
    ) -> None:
        """Build a service around a provider name, or an already-built adapter.

        Reading LLM_PROVIDER and LLM_MODEL from the environment means the eval
        scripts can be pointed at another provider without touching their code.
        """
        load_dotenv()
        if provider is None:
            provider = os.getenv("LLM_PROVIDER", DEFAULT_PROVIDER)
        if isinstance(provider, str):
            provider = registry.get(provider)()

        self.provider = provider
        # Left as None when unset so the provider's own default applies;
        # switching providers then does not also require knowing model names.
        self.model = model or os.getenv("LLM_MODEL") or None
        self.max_attempts = max_attempts
        self.backoff_seconds = backoff_seconds

    def chat(
        self,
        messages: list[Message],
        system: str | None = None,
        tools: list[ToolSchema] | None = None,
        text_format: type[BaseModel] | None = None,
        model: str | None = None,
    ) -> ChatResponse:
        """Call the provider with exponential-backoff retries.

        Pass text_format (a Pydantic model) for structured output; the parsed
        object arrives on the returned response as response.parsed. Pass tools
        for agentic tool calling, and system for the instruction — never as an
        entry in messages, since providers place it differently.

        The whole thing sits in one span, waiting included. Auto-instrumentation
        already times each attempt; what it cannot show is the sleeping between
        them, so a rate-limited call would otherwise read as an unexplained gap.
        """
        with tracer.start_as_current_span(
            "llm.chat", openinference_span_kind="chain"
        ) as span:
            last_error: Exception | None = None
            waited = 0.0

            for attempt in range(1, self.max_attempts + 1):
                try:
                    return self.provider.chat(
                        messages=messages,
                        system=system,
                        tools=tools,
                        text_format=text_format,
                        model=model or self.model,
                    )
                except Exception as error:
                    # The provider owns its error taxonomy; anything it does not
                    # call retryable is a caller bug and propagates immediately.
                    if not self.provider.is_retryable(error):
                        raise
                    last_error = error
                    if attempt == self.max_attempts:
                        break
                    delay = self.backoff_seconds * 2 ** (attempt - 1)
                    # Equal jitter: half the wait fixed, half random. Sub-agents
                    # fan out four at a time onto one provider, and a delay that
                    # is purely a function of the attempt number brings all four
                    # back on the same second to take the same rate limit again.
                    delay = delay / 2 + random.uniform(0, delay / 2)
                    span.add_event(
                        "retry",
                        {
                            "attempt": attempt,
                            "error": type(error).__name__,
                            "delay_s": round(delay, 3),
                        },
                    )
                    logger.warning(
                        "%s call failed (%s), attempt %d/%d, retrying in %.1fs",
                        type(self.provider).__name__,
                        type(error).__name__,
                        attempt,
                        self.max_attempts,
                        delay,
                    )
                    time.sleep(delay)
                    waited += delay
                finally:
                    # In a finally because the successful call leaves by return
                    # and never reaches the bottom of the loop: a fast call has
                    # to report a retry count of 0, not no count at all.
                    span.set_attribute("retry.count", attempt - 1)
                    span.set_attribute("retry.wait_seconds", round(waited, 3))

            raise last_error

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
