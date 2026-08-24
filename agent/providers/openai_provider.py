import base64
import json
import mimetypes
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import openai
from openai import OpenAI
from pydantic import BaseModel

from agent.tool_registry import ToolSchema

from .base import (
    ChatResponse,
    Message,
    ToolCall,
    UnsupportedFile,
    Usage,
)


# Point at the local MLX server rather than the OpenAI cloud. Either
# agent/providers/server.py (text only) or mlx_vlm.server serves this address;
# the latter is what multimodal input needs, since mlx-lm loads Gemma 4's text
# tower alone:
#     python -m mlx_vlm.server --model mlx-community/gemma-4-e2b-it-4bit \
#         --host 127.0.0.1 --port 8080 --max-kv-size 16384
# --host is passed explicitly because mlx_vlm.server defaults to 0.0.0.0, which
# would publish an unauthenticated model to the local network.
BASE_URL = "http://127.0.0.1:8080/v1"

# Attachment parts carry their source filename under this key. It is not part
# of the OpenAI schema; unknown keys
# on a content part are passed through as JSON and ignored by both the cloud
# API and mlx_vlm.server, which is what makes this cheaper than tracking
# filenames in a structure parallel to the history.
FILENAME_KEY = "x_filename"

# Which suffixes count as which kind of attachment. This table and the dispatch
# in _file_part() are the whole extension point: Gemma 4 has an audio tower and
# mlx_vlm.server reports video support, so adding either means adding a suffix
# set here plus one builder below -- and no signature between Agent.ask and
# here changes, which is the reason files= is one list rather than a keyword
# per kind.
IMAGE_SUFFIXES = frozenset(
    {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff"}
)

# That server never authenticates, but the SDK refuses to construct a client
# without a key. This is a placeholder, not a secret.
API_KEY = "not-needed"

# agent/providers/server.py ignores the `model` field entirely, but
# mlx_vlm.server does not: it treats the name as a model to load, and a name
# that differs from the loaded one evicts the weights and loads again -- about
# 12s per request. The HF repo id is lowercase `e2b` and the Hub only redirects
# the capitalized form, so the case here has to match what the server was
# started with or every request pays for a reload.
DEFAULT_MODEL = "mlx-community/gemma-4-e2b-it-4bit"

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
        messages: list[Message],
        system: str | None = None,
        tools: list[ToolSchema] | None = None,
        text_format: type[BaseModel] | None = None,
        model: str | None = None,
    ) -> ChatResponse:
        # Chat Completions takes the system prompt as an ordinary message, so
        # it is prepended to the translated copy here. Keeping it out of the
        # stored history is what lets Gemini, which passes it as config
        # instead, share one loop.
        payload = self._to_wire(messages)
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

    def _to_wire(self, messages: list[Message]) -> list[dict]:
        """Translate neutral Messages into Chat Completions items.

        The one direction that matters. Nothing above this file holds a dict in
        this shape any more, so a stored session can be replayed here or against
        any other adapter without being rewritten.
        """
        return [self._one(message) for message in messages]

    def _one(self, message: Message) -> dict:
        if message.role == "tool":
            result = message.tool_result
            return {
                "role": "tool",
                "tool_call_id": result.call_id,
                "content": result.output,
            }

        if message.role == "assistant":
            item: dict[str, Any] = {"role": "assistant", "content": message.text or None}
            if message.tool_calls:
                item["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        # Back to a JSON string: the wire format is a string
                        # even though ToolCall.arguments is parsed.
                        "function": {
                            "name": call.name,
                            "arguments": json.dumps(call.arguments),
                        },
                    }
                    for call in message.tool_calls
                ]
            return item

        # Content stays a plain string with no attachments. Chat Completions
        # accepts either that or a list of parts, and the string form keeps an
        # ordinary turn the same shape it has always been.
        if not message.attachments:
            return {"role": "user", "content": message.text}
        return {
            "role": "user",
            "content": [
                {"type": "text", "text": message.text},
                *(self._file_part(a.ref) for a in message.attachments),
            ],
        }

    @classmethod
    def _file_part(cls, file: str) -> dict:
        """Build one content part for an attachment, chosen by its suffix.

        Images only for now. Audio and video are what this dispatch exists to
        grow into: the model and mlx_vlm.server both handle them, so what is
        missing is a builder here, not a capability underneath.
        """
        suffix = cls._suffix_of(file)
        if suffix in IMAGE_SUFFIXES:
            return cls._image_part(file)

        raise UnsupportedFile(
            f"{file!r}: {suffix or 'no suffix'} is not a supported attachment. "
            f"This provider currently sends images only "
            f"({', '.join(sorted(IMAGE_SUFFIXES))})."
        )

    @staticmethod
    def _suffix_of(file: str) -> str:
        """Lowercased suffix of a path or URL.

        Parsed through urlparse so a query string does not end up inside the
        suffix — "photo.png?v=2" is still a PNG.
        """
        name = file if "://" not in file else urlparse(file).path
        return Path(name).suffix.lower()

    @staticmethod
    def _image_part(image: str) -> dict:
        """Build one image part from a local path, an http(s) URL, or a data URI.

        Local files are inlined as base64 rather than passed along as paths.
        mlx_vlm.server does accept a bare path and read the file itself, but
        only while it shares a filesystem with this process: the same history
        replayed against the OpenAI cloud, or against a server in the container
        from dockerfile, would fail or read a different file. Inlining costs
        ~33% in request size and makes the history self-contained.
        """
        if image.startswith(("http://", "https://", "data:")):
            return {"type": "image_url", "image_url": {"url": image}}

        path = Path(image)
        # Falls back to PNG rather than raising: a screenshot saved without an
        # extension is worth sending, and the server sniffs the payload anyway.
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
        encoded = base64.b64encode(path.read_bytes()).decode()
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{encoded}"},
            FILENAME_KEY: path.name,
        }

    def is_retryable(self, error: Exception) -> bool:
        return isinstance(error, RETRYABLE_ERRORS)

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
