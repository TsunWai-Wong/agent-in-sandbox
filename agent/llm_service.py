import logging
import os
import time
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel

from src.agent.tool_registry import ToolSchema

from .providers import ChatResponse, Provider, ToolResult, registry


logger = logging.getLogger(__name__)

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
        messages: list,
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
        """
        last_error: Exception | None = None

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
                logger.warning(
                    "%s call failed (%s), attempt %d/%d, retrying in %.1fs",
                    type(self.provider).__name__,
                    type(error).__name__,
                    attempt,
                    self.max_attempts,
                    delay,
                )
                time.sleep(delay)

        raise last_error

    def user_message(self, text: str) -> Any:
        """Build one user turn in the active provider's history format."""
        return self.provider.user_message(text)

    def extend(
        self,
        messages: list,
        response: ChatResponse,
        results: list[ToolResult],
    ) -> list:
        """Append a model turn, and any tool outputs, to the history."""
        return self.provider.extend(messages, response, results)

    def split_turns(
        self, messages: list, keep_last_turns: int
    ) -> tuple[list, list]:
        """Split history into (older, recent) at a user-turn boundary."""
        return self.provider.split_turns(messages, keep_last_turns)

    def compact(self, messages: list, keep_last_turns: int = 1) -> list:
        """Drop tool traffic from every turn but the most recent ones."""
        return self.provider.compact(messages, keep_last_turns)

    def render_transcript(self, messages: list) -> str:
        """Flatten history to plain text for the summarizer."""
        return self.provider.render_transcript(messages)
