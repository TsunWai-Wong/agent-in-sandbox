import json
import os
from typing import Any

import openai
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

from src.agent.tool_registry import ToolSchema

from .base import ChatResponse, ToolCall, ToolResult, Usage


DEFAULT_MODEL = "gpt-5.4-mini"

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
    """Adapter for the OpenAI Responses API."""

    client: OpenAI
    default_model: str

    def __init__(self, model: str | None = None) -> None:
        load_dotenv()
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Add it to your .env file."
            )
        # max_retries=0 disables the SDK's built-in retries so that the retry
        # loop in LLMService is the only one running.
        self.client = OpenAI(max_retries=0)
        self.default_model = model or DEFAULT_MODEL

    def chat(
        self,
        messages: list,
        system: str | None = None,
        tools: list[ToolSchema] | None = None,
        text_format: type[BaseModel] | None = None,
        model: str | None = None,
    ) -> ChatResponse:
        # The Responses API takes the system prompt as an ordinary message, so
        # it is prepended to a copy here. Keeping it out of the stored history
        # is what lets Gemini, which passes it as config instead, share one loop.
        payload = list(messages)
        if system is not None:
            payload.insert(0, {"role": "system", "content": system})

        kwargs: dict[str, Any] = {
            "model": model or self.default_model,
            "input": payload,
        }
        if tools is not None:
            kwargs["tools"] = [self._to_function_tool(tool) for tool in tools]

        if text_format is not None:
            response = self.client.responses.parse(
                text_format=text_format, **kwargs
            )
        else:
            response = self.client.responses.create(**kwargs)

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

        The Responses API accepts its own output items back verbatim, so the
        raw items are reused rather than rebuilt from the normalized response.
        """
        return [
            *messages,
            *response.raw.output,
            *(
                {
                    "type": "function_call_output",
                    "call_id": result.call_id,
                    "output": result.output,
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
        # Keeping only questions and answer text drops function_call,
        # function_call_output and reasoning items as a set. Dropping them
        # together is the point: a function_call_output whose call_id no
        # longer matches a function_call is a 400 on the next request.
        kept = [
            item
            for item in older
            if self._is_user_turn(item) or self._item_type(item) == "message"
        ]
        return [*kept, *recent]

    def render_transcript(self, messages: list) -> str:
        lines = []
        for item in messages:
            if self._is_user_turn(item):
                lines.append(f"User: {item['content']}")
            elif self._item_type(item) == "message":
                lines.append(f"Assistant: {self._message_text(item)}")
        return "\n\n".join(lines)

    def is_retryable(self, error: Exception) -> bool:
        return isinstance(error, RETRYABLE_ERRORS)

    @staticmethod
    def _item_type(item: Any) -> str | None:
        """Read an item's type, whether it is an SDK object or a plain dict.

        History holds both: output items come back as SDK objects, while the
        tool outputs extend() appends are dicts we built ourselves.
        """
        if isinstance(item, dict):
            return item.get("type")
        return getattr(item, "type", None)

    @staticmethod
    def _is_user_turn(item: Any) -> bool:
        """Whether this item is a real user question, not a tool result.

        Tool outputs carry "type" and no "role", so only the messages built by
        user_message() match — which is what makes this a safe cut point.
        """
        return isinstance(item, dict) and item.get("role") == "user"

    @staticmethod
    def _message_text(item: Any) -> str:
        """Join the text parts of an assistant message item."""
        return "".join(
            part.text
            for part in (getattr(item, "content", None) or [])
            if getattr(part, "text", None)
        )

    @staticmethod
    def _to_function_tool(schema: ToolSchema) -> dict:
        """Wrap a neutral schema in the Responses API function envelope."""
        # exclude_none, or every non-array parameter ships "items": null and
        # the API rejects the schema. This dump goes on the wire as JSON.
        return {"type": "function", **schema.model_dump(exclude_none=True)}

    @staticmethod
    def _to_chat_response(response: Any) -> ChatResponse:
        """Flatten the Responses output list into a ChatResponse."""
        text = None
        tool_calls = []

        for item in response.output:
            if item.type == "function_call":
                tool_calls.append(
                    ToolCall(
                        # call_id, not id: call_id is the one the API matches a
                        # function_call_output against.
                        id=item.call_id,
                        name=item.name,
                        arguments=json.loads(item.arguments or "{}"),
                    )
                )
            elif item.type == "message":
                text = item.content[0].text

        usage = getattr(response, "usage", None)

        return ChatResponse(
            text=text,
            tool_calls=tool_calls,
            usage=Usage(
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                total_tokens=usage.total_tokens,
            )
            if usage
            else None,
            # Only responses.parse() produces this attribute.
            parsed=getattr(response, "output_parsed", None),
            raw=response,
        )
