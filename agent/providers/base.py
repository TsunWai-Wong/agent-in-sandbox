"""Provider-neutral types shared by every model provider.

Application code — the agent loop, the evaluators — only ever sees the models
defined here. Translating them to and from a vendor SDK is the sole job of the
adapters in this package, so adding a provider never reaches past this module.
"""

from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from src.agent.tool_registry import ToolSchema


class ToolCall(BaseModel):
    """A model's request to run one tool."""

    model_config = ConfigDict(frozen=True)

    # Providers disagree on whether a call carries a correlation id: OpenAI
    # always sends one, Gemini usually omits it. Adapters synthesize an id when
    # it is missing, so callers never have to handle the absent case.
    id: str
    name: str
    # Always parsed. OpenAI hands back a JSON string and Gemini a dict; that
    # difference dies in the adapter rather than at every call site.
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """The outcome of running one ToolCall, on its way back to the model."""

    model_config = ConfigDict(frozen=True)

    # Both identifiers are carried because providers match a result to its call
    # differently: OpenAI keys off call_id, Gemini off the function name. The
    # normalized type holds the union of what providers need, not the overlap.
    call_id: str
    name: str
    output: str


class Usage(BaseModel):
    """Token counts for one model call, normalized across providers.

    Reported by the API rather than estimated locally, so the numbers the
    compaction thresholds compare against are the ones that were actually
    billed. Providers name these differently — OpenAI says input/output,
    Gemini says prompt/candidates — and that difference dies in the adapter.
    """

    model_config = ConfigDict(frozen=True)

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class ChatResponse(BaseModel):
    """One model turn, normalized across providers."""

    model_config = ConfigDict(frozen=True)

    text: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    # None when the provider did not report usage; callers must handle that
    # rather than assume a zero count means an empty prompt.
    usage: Usage | None = None
    # The structured-output instance, populated only when text_format was given.
    parsed: Any = None
    # The untouched provider response. Excluded from model_dump() so a
    # serialized ChatResponse stays JSON-clean for tracing; only the adapter
    # that produced it may read this.
    raw: Any = Field(default=None, exclude=True, repr=False)


class Provider(Protocol):
    """What LLMService needs from a model provider.

    Implementations translate and nothing else. Retries, backoff and logging
    live in LLMService so they are written once instead of once per provider.

    History is deliberately opaque: chat() and extend() take and return whatever
    shape the provider's SDK wants, and no caller inspects it. That keeps the
    agent loop provider-agnostic without a full Message type per provider, and
    it can be promoted to normalized messages later by changing only adapters.

    Context management is why split_turns(), compact() and render_transcript()
    exist here rather than in the agent: they are the only operations that must
    read that opaque history, so they are implemented once per format and the
    policy deciding when to call them stays provider-agnostic in Conversation.
    """

    default_model: str

    def chat(
        self,
        messages: list,
        system: str | None = None,
        tools: list[ToolSchema] | None = None,
        text_format: type[BaseModel] | None = None,
        model: str | None = None,
    ) -> ChatResponse:
        """Send one turn and return the normalized response.

        The system prompt is a separate argument rather than an entry in
        messages because providers place it differently: OpenAI takes it as an
        ordinary message, Gemini as system_instruction on the request config.
        """
        ...

    def user_message(self, text: str) -> Any:
        """Build one user turn in this provider's history format."""
        ...

    def extend(
        self,
        messages: list,
        response: ChatResponse,
        results: list[ToolResult],
    ) -> list:
        """Append a model turn, and any tool outputs, to the history."""
        ...

    def split_turns(
        self, messages: list, keep_last_turns: int
    ) -> tuple[list, list]:
        """Split history into (older, recent) at a user-turn boundary.

        Only a real user question starts a turn — never a tool result, which
        some providers also send under the user role. Cutting anywhere else
        would separate a tool call from its result and make the next request
        invalid, so this is the only place a split may happen.
        """
        ...

    def compact(self, messages: list, keep_last_turns: int = 1) -> list:
        """Drop tool traffic from every turn but the most recent ones.

        Older turns keep only the question and the answer text. Removing whole
        items rather than parts of them is what makes orphaned call/result
        pairs impossible instead of merely unlikely.
        """
        ...

    def render_transcript(self, messages: list) -> str:
        """Flatten history to plain text for the summarizer.

        Only ever runs on already-compacted history, so it sees questions and
        answers and never has to render tool traffic.
        """
        ...

    def is_retryable(self, error: Exception) -> bool:
        """Whether LLMService should retry after this error.

        A predicate rather than a tuple of exception types: OpenAI raises a
        distinct class per failure mode, while google-genai raises one APIError
        carrying a status code, which no isinstance check can separate.
        """
        ...
