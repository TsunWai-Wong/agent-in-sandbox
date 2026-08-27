"""Provider-neutral types shared by every model provider.

Application code — the agent loop, the evaluators — only ever sees the models
defined here. Translating them to and from a vendor SDK is the sole job of the
adapters in this package, so adding a provider never reaches past this module.
"""

from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from agent.tool_registry import ToolSchema


# Seconds before a request is abandoned. Defined here rather than on either
# adapter because LLMService needs the same number when it builds a chain, and
# importing it from an adapter would point the dependency the wrong way.
DEFAULT_TIMEOUT = 60.0


class UnsupportedFile(ValueError):
    """Raised for an attachment this provider cannot put on a message.

    Raised while building the turn, before anything reaches the network, so an
    unsupported file fails on the line that named it rather than as a 400 from
    the server several frames later.
    """


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


class Attachment(BaseModel):
    """A file on a user turn: a local path or a URL, plus what kind it is.

    The reference travels, never the bytes. Providers disagree on how to carry
    an attachment — OpenAI inlines a data URI, Gemini uploads to its Files API
    and references a URI — so the adapter resolves this at send time. Storing
    the reference is also what keeps an event JSON-small enough to persist.
    """

    model_config = ConfigDict(frozen=True)

    ref: str
    kind: str = "image"


class Message(BaseModel):
    """One turn of history, in the shape every provider translates from.

    This is the neutral format the Provider docstring below promised. History
    used to be opaque provider dicts; it is these now, so a session recorded
    against one provider can be replayed against another, and an adapter is
    left with translation and nothing else.

    Four roles' worth of content in one model rather than a class per role: the
    combinations are exclusive in practice, and a union would put an isinstance
    check in every adapter and every filter for no gain.
    """

    model_config = ConfigDict(frozen=True)

    # "user" | "assistant" | "tool". Matches Event.kind one-to-one, which is
    # what lets the store answer "is this history?" without parsing content.
    role: str
    text: str = ""
    attachments: list[Attachment] = Field(default_factory=list)
    # Assistant turns only: what the model asked to run.
    tool_calls: list[ToolCall] = Field(default_factory=list)
    # Tool turns only: the answer to one of those calls. Carries call_id, so a
    # provider that pairs on id and one that pairs on name are both satisfied.
    tool_result: ToolResult | None = None


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

    Implementations translate and nothing else. Retries, fallback, backoff and
    logging live in LLMService so they are written once instead of once per
    provider.

    Two members, down from eight. History is neutral Messages now, so building
    a user turn, appending a model turn, splitting at a turn boundary and
    rendering a transcript are all provider-independent and moved out — to the
    event store, which records them, and to ContextAssembler, which shapes
    them. Judging errors went the same way: providers/errors.py reads them for
    the whole package, so is_retryable is gone too. What is left here is
    genuinely per-vendor: the wire format.

    Constructors take `model` and `timeout` and are called with nothing else,
    since LLMService builds one adapter per entry in its chain.

    That is the whole reason for the neutral type. Adding mlx, vllm or
    OpenRouter is now a wire translation and nothing else.
    """

    default_model: str

    def chat(
        self,
        messages: list[Message],
        system: str | None = None,
        tools: list[ToolSchema] | None = None,
        text_format: type[BaseModel] | None = None,
        model: str | None = None,
    ) -> ChatResponse:
        """Send one turn and return the normalized response.

        Translating Messages to the wire shape happens inside, so no caller
        ever holds a provider-specific dict. The system prompt is a separate
        argument rather than an entry in messages because providers place it
        differently: OpenAI takes it as an ordinary message, Gemini as
        system_instruction on the request config.

        Raises UnsupportedFile for an attachment kind this provider cannot send.
        """
        ...
