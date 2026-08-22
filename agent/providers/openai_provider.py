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

from .base import ChatResponse, ToolCall, ToolResult, UnsupportedFile, Usage


# Point at the local MLX server rather than the OpenAI cloud. Either
# agent/providers/server.py (text only) or mlx_vlm.server serves this address;
# the latter is what multimodal input needs, since mlx-lm loads Gemma 4's text
# tower alone:
#     python -m mlx_vlm.server --model mlx-community/gemma-4-e2b-it-4bit \
#         --host 127.0.0.1 --port 8080 --max-kv-size 16384
# --host is passed explicitly because mlx_vlm.server defaults to 0.0.0.0, which
# would publish an unauthenticated model to the local network.
BASE_URL = "http://127.0.0.1:8080/v1"

# Attachment parts carry their source filename under this key so that compact()
# can name what it stripped. It is not part of the OpenAI schema; unknown keys
# on a content part are passed through as JSON and ignored by both the cloud
# API and mlx_vlm.server, which is what makes this cheaper than tracking
# filenames in a structure parallel to the history.
FILENAME_KEY = "x_filename"

# Which suffixes count as which kind of attachment. This table and the dispatch
# in _file_part() are the whole extension point: Gemma 4 has an audio tower and
# mlx_vlm.server reports video support, so adding either means adding a suffix
# set here plus one builder below -- and no signature between Conversation.ask
# and here changes, which is the reason files= is one list rather than a
# keyword per kind.
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

    def user_message(self, text: str, files: list[str] | None = None) -> dict:
        """Build one user turn, with any attachments included.

        Content stays a plain string when there are no files. Chat Completions
        accepts either that or a list of parts, and the string form is what
        every text-only path in this file already reads — keeping it means
        attachments do not change the shape of an ordinary turn.
        """
        if not files:
            return {"role": "user", "content": text}

        return {
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                *(self._file_part(file) for file in files),
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
        #
        # Images survive as a placeholder rather than being dropped with the
        # tool traffic: "what did I show you?" is a fair question three turns
        # later, and an answer referring to an image the history no longer
        # mentions at all reads as a hallucination.
        kept = [
            self._strip_images(item)
            for item in older
            if self._is_user_turn(item) or self._is_answer(item)
        ]
        return [*kept, *recent]

    @classmethod
    def _strip_images(cls, item: dict) -> dict:
        """Replace image parts with a text placeholder naming the file.

        The server is stateless, so every image in the history is re-sent and
        re-prefilled on every subsequent request — ~280 tokens each for Gemma 4
        E2B (config.json: vision_soft_tokens_per_image), plus its base64 in
        this process's memory for as long as the turn survives. Past the recent
        window the model is answering *about* the image rather than looking at
        it again, so the name earns its place and the pixels do not.

        Only the parts of a turn change, never its role or position, so
        split_turns() finds the same boundaries on the next pass.
        """
        content = item.get("content")
        # A text-only turn is already a bare string, which is the common case.
        if not isinstance(content, list):
            return item

        parts = [
            {"type": "text", "text": f"[image: {cls._image_name(part)}]"}
            if cls._is_image_part(part)
            else part
            for part in content
        ]
        # Rebuilt rather than mutated: the caller still holds this dict, and
        # compact() returning a new history while quietly editing the old one
        # would make the two disagree.
        return {**item, "content": parts}

    @staticmethod
    def _is_image_part(part: Any) -> bool:
        return isinstance(part, dict) and part.get("type") == "image_url"

    @staticmethod
    def _image_name(part: dict) -> str:
        """Name an image part for its placeholder.

        Prefers the filename recorded by _image_part(). A part that came from a
        URL keeps its basename; a data URI carries no name at all, so those
        degrade to a generic label rather than an empty one.
        """
        name = part.get(FILENAME_KEY)
        if name:
            return name

        url = (part.get("image_url") or {}).get("url", "")
        if url.startswith("data:"):
            return "image"
        return Path(urlparse(url).path).name or "image"

    def render_transcript(self, messages: list) -> str:
        lines = []
        for item in messages:
            if self._is_user_turn(item):
                lines.append(f"User: {self._as_text(item)}")
            elif self._is_answer(item):
                lines.append(f"Assistant: {self._as_text(item)}")
        return "\n\n".join(lines)

    @classmethod
    def _as_text(cls, item: dict) -> str:
        """Flatten one message's content to plain text.

        Interpolating multimodal content straight into an f-string would hand
        the summarizer the repr of a list of dicts — with a full base64 payload
        inside it. That is unreadable, and at roughly 1.4x the file size in
        characters it would blow the summarizer's own context on a single
        screenshot.
        """
        content = item.get("content")
        if content is None:
            return ""
        if not isinstance(content, list):
            return str(content)

        pieces = []
        for part in content:
            if not isinstance(part, dict):
                continue
            if cls._is_image_part(part):
                pieces.append(f"[image: {cls._image_name(part)}]")
            elif part.get("type") == "text":
                pieces.append(part.get("text", ""))
        return " ".join(piece for piece in pieces if piece)

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
