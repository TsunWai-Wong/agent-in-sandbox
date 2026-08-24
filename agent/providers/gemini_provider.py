"""Adapter for the Google Gemini API.

Written as the test of whether the neutral Message type earned its keep: this
file translates, and nothing above it changed to accommodate it.

Gemini disagrees with Chat Completions on nearly every surface detail — the
assistant is called "model", the system prompt is config rather than a message,
tool results come back under the user role, and they are paired to their call by
*name* rather than by id. All four differences die here.
"""

from __future__ import annotations

import json
import mimetypes
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from pydantic import BaseModel

from agent.tool_registry import ToolSchema

from .base import ChatResponse, Message, ToolCall, UnsupportedFile, Usage


DEFAULT_MODEL = "gemini-2.5-flash"

IMAGE_SUFFIXES = frozenset(
    {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff"}
)

# Gemini raises one APIError carrying a status code rather than a class per
# failure mode, so retryability is a code check rather than an isinstance —
# which is exactly why the protocol asks for a predicate and not a tuple.
RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})


class GeminiProvider:
    """Translate neutral Messages to and from google-genai."""

    default_model: str

    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        # Imported here, not at module scope: registry.py registers this class
        # at import time, and a top-level import would make the whole agent
        # package unimportable for anyone who has not installed the SDK.
        from google import genai

        self.client = genai.Client(api_key=api_key) if api_key else genai.Client()
        self.default_model = model or DEFAULT_MODEL

    def chat(
        self,
        messages: list[Message],
        system: str | None = None,
        tools: list[ToolSchema] | None = None,
        text_format: type[BaseModel] | None = None,
        model: str | None = None,
    ) -> ChatResponse:
        from google.genai import types

        config: dict[str, Any] = {}
        if system is not None:
            # Config, not a message — the reason the protocol keeps `system` a
            # separate argument instead of letting callers prepend a turn.
            config["system_instruction"] = system
        if tools is not None:
            config["tools"] = [
                types.Tool(
                    function_declarations=[
                        self._to_function_declaration(tool) for tool in tools
                    ]
                )
            ]
        if text_format is not None:
            config["response_mime_type"] = "application/json"
            config["response_schema"] = text_format

        response = self.client.models.generate_content(
            model=model or self.default_model,
            contents=self._to_wire(messages),
            config=types.GenerateContentConfig(**config) if config else None,
        )
        return self._to_chat_response(response)

    # -- outbound ------------------------------------------------------------

    def _to_wire(self, messages: list[Message]) -> list[dict]:
        """Translate neutral Messages into Gemini `contents`.

        Consecutive same-role turns are merged, which Chat Completions never
        needs and Gemini always does — it validates strict user/model
        alternation. Three ordinary things produce a run of same-role turns:
        the answers to a parallel tool batch (all function_responses, all under
        the user role), an Inject from a guardrail landing right after the
        user's question, and a conversation summary immediately followed by the
        first turn it did not cover.

        Merging here rather than upstream is deliberate. None of those runs is
        malformed — OpenAI accepts every one of them — so this is a fact about
        Gemini's wire format, and the neutral history stays neutral.
        """
        contents: list[dict] = []
        for message in messages:
            item = self._one(message)
            if contents and contents[-1]["role"] == item["role"]:
                contents[-1]["parts"].extend(item["parts"])
            else:
                contents.append(item)
        return contents

    def _one(self, message: Message) -> dict:
        if message.role == "tool":
            result = message.tool_result
            return {
                "role": "user",
                "parts": [
                    {
                        "function_response": {
                            # By NAME, where Chat Completions pairs by id. This
                            # is why ToolResult carries both: the neutral type
                            # holds the union of what providers need, not the
                            # overlap, so neither adapter has to invent one.
                            "name": result.name,
                            "response": {"output": result.output},
                        }
                    }
                ],
            }

        if message.role == "assistant":
            parts: list[dict] = []
            if message.text:
                parts.append({"text": message.text})
            parts.extend(
                {"function_call": {"name": call.name, "args": call.arguments}}
                for call in message.tool_calls
            )
            # Never an empty parts list: Gemini rejects a content with no parts,
            # and a pure tool-call turn has no text at all.
            return {"role": "model", "parts": parts or [{"text": ""}]}

        return {
            "role": "user",
            "parts": [
                {"text": message.text},
                *(self._file_part(a.ref) for a in message.attachments),
            ],
        }

    @classmethod
    def _file_part(cls, file: str) -> dict:
        """Inline an attachment as bytes.

        A URL is not fetched here: Gemini takes a file_uri only for something
        already uploaded to its Files API, and silently accepting an arbitrary
        http URL would make this adapter a fetcher of whatever a tool result
        happened to name.
        """
        name = file if "://" not in file else urlparse(file).path
        suffix = Path(name).suffix.lower()
        if suffix not in IMAGE_SUFFIXES:
            raise UnsupportedFile(
                f"{file!r} is not a kind this provider can send "
                f"({', '.join(sorted(IMAGE_SUFFIXES))})."
            )
        if file.startswith(("http://", "https://")):
            raise UnsupportedFile(
                f"{file!r} is a URL; Gemini takes uploaded files or local paths."
            )
        mime = mimetypes.guess_type(name)[0] or "image/png"
        return {"inline_data": {"mime_type": mime, "data": Path(file).read_bytes()}}

    @staticmethod
    def _to_function_declaration(schema: ToolSchema) -> dict:
        # Flat, where Chat Completions nests the same fields under "function".
        return schema.model_dump(exclude_none=True)

    # -- inbound -------------------------------------------------------------

    @staticmethod
    def _to_chat_response(response: Any) -> ChatResponse:
        parts = response.candidates[0].content.parts or []

        texts = [p.text for p in parts if getattr(p, "text", None)]
        tool_calls = [
            ToolCall(
                # Synthesized: Gemini sends no call id, and every caller above
                # this line is allowed to assume there is one. Deriving it from
                # the name and arguments keeps it stable if the same turn is
                # translated twice.
                id=f"gemini-{index}-{abs(hash(json.dumps(dict(p.function_call.args), sort_keys=True, default=str)))}",
                name=p.function_call.name,
                arguments=dict(p.function_call.args or {}),
            )
            for index, p in enumerate(parts)
            if getattr(p, "function_call", None)
        ]

        meta = getattr(response, "usage_metadata", None)
        return ChatResponse(
            text="".join(texts) or None,
            tool_calls=tool_calls,
            usage=Usage(
                # prompt/candidates, where Chat Completions says
                # prompt/completion and the Responses API said input/output.
                input_tokens=getattr(meta, "prompt_token_count", 0) or 0,
                output_tokens=getattr(meta, "candidates_token_count", 0) or 0,
                total_tokens=getattr(meta, "total_token_count", 0) or 0,
            )
            if meta
            else None,
            parsed=getattr(response, "parsed", None),
            raw=response,
        )

    def is_retryable(self, error: Exception) -> bool:
        return getattr(error, "code", None) in RETRYABLE_STATUS
