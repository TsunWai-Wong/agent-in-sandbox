import json
from typing import Any

import openai
from openai import OpenAI
from pydantic import BaseModel

from agent.tool_registry import ToolSchema

from .base import ChatResponse, ToolCall, ToolResult, Usage


# Point at the local MLX server in agent/providers/server.py rather than the
# OpenAI cloud.
BASE_URL = "http://127.0.0.1:8080/v1"

# That server never authenticates, but the SDK refuses to construct a client
# without a key. This is a placeholder, not a secret.
API_KEY = "not-needed"

# server.py serves whichever model it was started with and ignores the `model`
# field on the request entirely -- it is sent because the SDK requires one, and
# named to match server.py's own default so traces read honestly.
DEFAULT_MODEL = "mlx-community/gemma-4-E2B-it-4bit"

# The SDK raises typed exceptions instead of surfacing HTTP status codes:
# 429 -> RateLimitError, 5xx -> InternalServerError, and network problems or
# timeouts -> APIConnectionError. Anything else (401 auth, 400 bad request,
# 404 model not found) is a caller bug and must not be retried.
RETRYABLE_ERRORS = (
    openai.RateLimitError,
    openai.InternalServerError,
    openai.APIConnectionError,
)


class OpenAIProvider:
    """Adapter for the OpenAI Chat Completions API.

    Chat Completions rather than Responses because the target is the local
    server in agent/providers/server.py, which serves /v1/chat/completions and
    /v1/models and nothing else -- a Responses call 404s against it. It is also
    the interface every local runtime implements, so the same adapter works
    against mlx, vLLM or llama.cpp unchanged.
    """

    client: OpenAI
    default_model: str

    def __init__(self, model: str | None = None) -> None:
        # max_retries=0 disables the SDK's built-in retries so that the retry
        # loop in LLMService is the only one running.
        self.client = OpenAI(
            base_url=BASE_URL,
            api_key=API_KEY,
            max_retries=0,
        )
        self.default_model = model or DEFAULT_MODEL

    def chat(
        self,
        messages: list,
        system: str | None = None,
        tools: list[ToolSchema] | None = None,
        text_format: type[BaseModel] | None = None,
        model: str | None = None,
    ) -> ChatResponse:
        # Chat Completions takes the system prompt as an ordinary message, so
        # it is prepended to a copy here. Keeping it out of the stored history
        # is what lets Gemini, which passes it as config instead, share one loop.
        payload = list(messages)
        if system is not None:
            payload.insert(0, {"role": "system", "content": system})

        kwargs: dict[str, Any] = {
            "model": model or self.default_model,
            "messages": payload,
        }
        if tools is not None:
            kwargs["tools"] = [self._to_function_tool(tool) for tool in tools]

        if text_format is not None:
            # parse() sends the model as a json_schema response_format and
            # validates the reply back into it. Against server.py that schema
            # drives grammar-constrained decoding, so the reply is schema-valid
            # by construction rather than by luck.
            response = self.client.chat.completions.parse(
                response_format=text_format, **kwargs
            )
        else:
            response = self.client.chat.completions.create(**kwargs)

        return self._to_chat_response(response)

    def user_message(self, text: str) -> dict:
        return {"role": "user", "content": text}

    def extend(
        self,
        messages: list,
        response: ChatResponse,
        results: list[ToolResult],
    ) -> list:
        """Append the model turn and its tool outputs to the history.

        The assistant turn is rebuilt from the normalized response rather than
        reusing the SDK object, so history stays plain dicts. It goes back on
        the wire on the next request and is what render_transcript reads, and
        both of those want JSON, not SDK models.
        """
        assistant: dict[str, Any] = {
            "role": "assistant",
            "content": response.text,
        }
        if response.tool_calls:
            assistant["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        # Back to a JSON string: the wire format is a string
                        # even though ToolCall.arguments is parsed.
                        "arguments": json.dumps(call.arguments),
                    },
                }
                for call in response.tool_calls
            ]

        return [
            *messages,
            assistant,
            *(
                {
                    "role": "tool",
                    # tool_call_id is what pairs a result with its call. One
                    # that matches no tool_call is a 400 on the next request.
                    "tool_call_id": result.call_id,
                    "content": result.output,
                }
                for result in results
            ),
        ]

    def split_turns(
        self, messages: list, keep_last_turns: int
    ) -> tuple[list, list]:
        if keep_last_turns <= 0:
            return list(messages), []

        boundaries = [
            index
            for index, item in enumerate(messages)
            if self._is_user_turn(item)
        ]
        if len(boundaries) <= keep_last_turns:
            return [], list(messages)

        cut = boundaries[-keep_last_turns]
        return list(messages[:cut]), list(messages[cut:])

    def compact(self, messages: list, keep_last_turns: int = 1) -> list:
        older, recent = self.split_turns(messages, keep_last_turns)
        # Keeping only questions and answer text drops each assistant turn that
        # carried tool_calls together with the role="tool" results answering
        # it. Dropping them as a set is the point: a tool message whose
        # tool_call_id no longer matches a tool_call is a 400 on the next
        # request.
        kept = [
            item
            for item in older
            if self._is_user_turn(item) or self._is_answer(item)
        ]
        return [*kept, *recent]

    def render_transcript(self, messages: list) -> str:
        lines = []
        for item in messages:
            if self._is_user_turn(item):
                lines.append(f"User: {item['content']}")
            elif self._is_answer(item):
                lines.append(f"Assistant: {item.get('content') or ''}")
        return "\n\n".join(lines)

    def is_retryable(self, error: Exception) -> bool:
        return isinstance(error, RETRYABLE_ERRORS)

    @staticmethod
    def _is_user_turn(item: Any) -> bool:
        """Whether this item is a real user question, not a tool result.

        Chat Completions files tool results under role "tool", never "user", so
        only the messages built by user_message() match — which is what makes
        this a safe cut point.
        """
        return isinstance(item, dict) and item.get("role") == "user"

    @staticmethod
    def _is_answer(item: Any) -> bool:
        """Whether this item is a final assistant answer, not a tool call.

        An assistant turn carrying tool_calls is half of a pair; it is only
        valid in history while the tool messages answering it are still there,
        so compact() and the transcript both treat it as tool traffic.
        """
        return (
            isinstance(item, dict)
            and item.get("role") == "assistant"
            and not item.get("tool_calls")
        )

    @staticmethod
    def _to_function_tool(schema: ToolSchema) -> dict:
        """Wrap a neutral schema in the Chat Completions function envelope.

        Chat Completions nests the declaration under "function"; the Responses
        API took the same fields flat. That is the one shape difference between
        the two that reaches this far.
        """
        # exclude_none, or every non-array parameter ships "items": null and
        # the API rejects the schema. This dump goes on the wire as JSON.
        return {
            "type": "function",
            "function": schema.model_dump(exclude_none=True),
        }

    @staticmethod
    def _to_chat_response(response: Any) -> ChatResponse:
        """Flatten one Chat Completions choice into a ChatResponse."""
        message = response.choices[0].message

        tool_calls = [
            ToolCall(
                id=call.id,
                name=call.function.name,
                arguments=json.loads(call.function.arguments or "{}"),
            )
            for call in (message.tool_calls or [])
        ]

        usage = getattr(response, "usage", None)

        return ChatResponse(
            # None on a pure tool-call turn, which is why callers check
            # tool_calls first rather than treating empty text as an answer.
            text=message.content,
            tool_calls=tool_calls,
            usage=Usage(
                # Chat Completions says prompt/completion where Responses said
                # input/output; the neutral Usage keeps the latter names.
                input_tokens=usage.prompt_tokens,
                output_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
            )
            if usage
            else None,
            # Only chat.completions.parse() populates this.
            parsed=getattr(message, "parsed", None),
            raw=response,
        )
