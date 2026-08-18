#!/usr/bin/env python3
"""
agent/providers/server.py
A small OpenAI-compatible server for Gemma 4 E2B on Apple silicon via MLX.

WHAT IT DOES
------------
POST /v1/chat/completions

  "tools": [...]
      Native Gemma 4 tool calling. The model is prompted through its own chat
      template, and the reply is parsed with mlx-lm's built-in gemma4 parser.
      If that parse fails, the request is retried with constrained decoding so
      you always get a well-formed tool call back.

  "response_format": {"type": "json_schema",
                      "json_schema": {"schema": {...}}}
      Grammar-constrained decoding. The output cannot violate the schema.
      Pydantic works client-side: MyModel.model_json_schema().

WHY THIS SHAPE
--------------
Tool calling and structured output are the same problem: force the decoder to
only emit tokens a grammar allows. One model in memory, two entry points.

  - Unconstrained path -> mlx_lm.generate       (fast, in-distribution)
  - Constrained path   -> outlines.from_mlxlm   (guaranteed valid)

Outlines wraps the *already loaded* mlx-lm model object, so there is only ever
one copy of the weights in RAM. That matters on an 8 GB Air.

WHY A CLASS
-----------
Everything below hangs off `Server`, whose __init__ loads the weights. As a
module of top-level statements this file loaded a multi-gigabyte model as a
side effect of being imported, which is tolerable for a script run directly and
not tolerable for a module sitting inside a package -- `import agent.providers`
would have paid for it. Constructing a Server is now the explicit act that
costs RAM.

INSTALL (macOS 15+, Apple silicon)
----------------------------------
    pip install "mlx-lm>=0.31.3" "outlines>=1.0" fastapi uvicorn pydantic

    Pin these. Both mlx-lm's sampling API and Outlines' top-level API have
    changed shape between releases.

RUN
---
    python agent/providers/server.py
    # or, equivalently:
    python -m agent.providers.server
    # -> http://127.0.0.1:8080/v1

THREE SPOTS TO VERIFY AGAINST YOUR INSTALLED VERSIONS
-----------------------------------------------------
Marked [VERIFY] in the code below. They are the version-sensitive bits:
  1. the tool-parser import path
  2. the Outlines raw-JSON-schema helper
  3. the chat template's thinking-mode kwarg
"""

import asyncio
import json
import os
import threading
import time
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

import mlx_lm
import outlines
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from mlx_lm.sample_utils import make_sampler
from pydantic import BaseModel

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

# QAT 4-bit: Google trained these weights to survive 4-bit, so quality is
# noticeably better than a plain post-hoc 4-bit quant at the same size.
# Alternatives: mlx-community/gemma-4-e2b-it-4bit
#               lmstudio-community/gemma-4-E2B-it-MLX-4bit
# Overridable so one process can be pointed at one model: benchmarking.py
# starts a fresh server per model rather than trying to hold several sets of
# weights in 8 GB at once.
MODEL_ID = os.environ.get("MODEL_ID", "mlx-community/gemma-4-E2B-it-4bit")

# E2B advertises a 128K window. Do NOT let the KV cache grow to that on an Air.
# A rotating cache of 4096 keeps memory flat; raise it if you need more recall.
MAX_KV_SIZE = 4096

DEFAULT_MAX_TOKENS = 1024

# Google's recommended sampling for Gemma 4 chat.
CHAT_SAMPLING = dict(temp=1.0, top_p=0.95, top_k=64)

# For schema-constrained output you want the model boring and obedient.
# The grammar handles validity; sampling entropy only buys you worse values.
STRUCTURED_SAMPLING = dict(temp=0.0)

# PORT is env-overridable too, so a benchmark sweep can use its own port
# instead of fighting a server you started by hand on 8080.
HOST, PORT = "127.0.0.1", int(os.environ.get("PORT", 8080))


# [VERIFY 1] Tool parser location. In mlx-lm 0.31.x this is:
#     mlx_lm/tool_parsers/gemma4.py :: parse_tool_call(text, tools)
# It is semi-internal and has moved before. Check with:
#     python -c "import mlx_lm.tool_parsers as t; print(dir(t))"
try:
    from mlx_lm.tool_parsers import gemma4 as gemma4_parser
except ImportError:
    gemma4_parser = None
    print("warning: gemma4 tool parser not found; tool calls will always "
          "fall back to constrained decoding")


# --------------------------------------------------------------------------
# Request models
# --------------------------------------------------------------------------

class ChatRequest(BaseModel):
    model: Optional[str] = None
    messages: List[Dict[str, Any]]
    tools: Optional[List[Dict[str, Any]]] = None
    response_format: Optional[Dict[str, Any]] = None
    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: Optional[float] = None
    # Declare `stream` even though the non-streaming path ignores it. Pydantic
    # drops unknown keys silently, so without this field a client asking for
    # SSE got a single JSON body back and no error -- which is exactly how
    # benchmark harnesses end up reporting a null time-to-first-token.
    stream: bool = False
    stream_options: Optional[Dict[str, Any]] = None
    # Gemma 4 has configurable thinking. Off by default here: a reasoning
    # preamble and a strict JSON grammar interact badly (see notes at bottom).
    thinking: bool = False


# --------------------------------------------------------------------------
# Agent
# --------------------------------------------------------------------------

class Server:
    """One loaded model, served over an OpenAI-compatible FastAPI app.

    Usage:
        server = Server()        # loads the weights
        server.run()             # serves them on HOST:PORT

    Construct this on the main thread. MLX streams are thread-local, so the
    thread that loads the model is the only one that may generate from it --
    see the note on `async def` in chat_completions().
    """

    def __init__(self,
                 model_id: str = MODEL_ID,
                 *,
                 max_kv_size: int = MAX_KV_SIZE,
                 host: str = HOST,
                 port: int = PORT):
        self.model_id = model_id
        self.max_kv_size = max_kv_size
        self.host = host
        self.port = port

        # ---- Load once, share everywhere --------------------------------
        print(f"loading {model_id} ...")
        self.model, self.tokenizer = mlx_lm.load(model_id)

        # Register the tokenizer's own declared EOS. mlx-lm seeds its stop set
        # from the chat template's end-of-turn token, which is not always
        # eos_token -- Phi-4-mini ends turns with <|end|> but declares
        # <|endoftext|>. When such a model emits its eos_token nothing stops
        # generation: it runs all the way to max_tokens and the special tokens
        # are detokenized into the reply as literal text, e.g.
        #     { "answer": "B" }<|endoftext|><|endoftext|>...
        # which breaks any client calling model_validate_json() on the result.
        if getattr(self.tokenizer, "eos_token", None):
            self.tokenizer.add_eos_token(self.tokenizer.eos_token)

        # Same weights, second interface. No extra RAM.
        self.constrained_model = outlines.from_mlxlm(self.model, self.tokenizer)

        # MLX generation is single-stream. Serialize requests or you will get
        # interleaved garbage and confusing Metal errors under concurrency.
        self.gpu_lock = threading.Lock()

        # Serializes whole requests, streaming ones included. This has to be an
        # asyncio.Lock rather than gpu_lock: a streaming response hands control
        # back to the event loop between chunks, and a threading.Lock held
        # across a yield would block the one thread that could ever release it
        # -- a hard deadlock. gpu_lock still guards the synchronous generators;
        # under this lock it is always uncontended.
        self.request_lock = asyncio.Lock()

        self.app = self._build_app()

    # ---------------------------------------------------------------- app

    def _build_app(self) -> FastAPI:
        """Wire the routes to this instance.

        The handlers are thin closures over bound methods so that the routing
        layer holds no state of its own: two Servers in one process would get
        two independent apps rather than fighting over module globals.
        """
        app = FastAPI(title="gemma4-mlx")

        @app.post("/v1/chat/completions")
        async def chat_completions(req: ChatRequest):
            return await self.chat_completions(req)

        @app.get("/v1/models")
        def list_models():
            return self.list_models()

        return app

    def run(self) -> None:
        # workers=1 is deliberate. One GPU stream, one process.
        uvicorn.run(self.app, host=self.host, port=self.port, workers=1)

    # ---------------------------------------------------------- prompting

    def build_prompt(self,
                     messages: List[Dict[str, Any]],
                     tools: Optional[List[Dict[str, Any]]] = None,
                     thinking: bool = False) -> str:
        """Render messages through Gemma 4's own chat template.

        Passing `tools` here is what puts the model in-distribution for tool
        calling -- the template injects the schemas in the exact format the
        model saw during training. Do not hand-roll this into a system prompt.
        """
        kwargs: Dict[str, Any] = dict(
            add_generation_prompt=True,
            tokenize=False,
        )
        if tools:
            kwargs["tools"] = tools

        # The conversation is positional. HF names that parameter
        # `conversation`, not `messages`, and mlx-lm's TokenizerWrapper
        # forwards *args through untouched -- so passing messages=... raises
        # "missing 1 required positional argument: 'conversation'".
        #
        # [VERIFY 3] resolved: mlx-lm's wrapper takes `enable_thinking` and
        # defaults it to tokenizer.has_thinking, so this kwarg is always safe.
        return self.tokenizer.apply_chat_template(
            self._decode_tool_arguments(messages), **kwargs,
            enable_thinking=thinking)

    @staticmethod
    def _decode_tool_arguments(
        messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Parse assistant tool-call arguments from the wire format to dicts.

        OpenAI puts `arguments` on a tool call as a JSON *string*, and every
        OpenAI-compatible client sends it that way. Gemma's chat template
        expects a mapping and braces it itself, so a string arrives already
        braced and renders doubled:

            {{"city": "Frankfurt"}}   instead of   {city:<|"|>Frankfurt<|"|>}

        That is out of distribution, and the model answers the following turn
        with an immediate end-of-turn instead of reading the tool result --
        a multi-turn tool loop returns empty text rather than an answer.
        Decoding here keeps the HTTP surface OpenAI-compatible while the
        prompt stays in the shape the model was trained on.
        """
        out: List[Dict[str, Any]] = []
        for message in messages:
            calls = message.get("tool_calls") if isinstance(message, dict) else None
            if not calls:
                out.append(message)
                continue

            decoded = []
            for call in calls:
                fn = call.get("function", {})
                args = fn.get("arguments")
                if not isinstance(args, str):
                    decoded.append(call)
                    continue
                try:
                    parsed = json.loads(args or "{}")
                except json.JSONDecodeError:
                    # Render a malformed string as-is rather than dropping the
                    # call: a gap in the history confuses the model more than
                    # an ugly argument does.
                    decoded.append(call)
                    continue
                decoded.append({**call, "function": {**fn, "arguments": parsed}})

            # Rebuilt rather than mutated -- these dicts came off the request
            # and are not ours to edit in place.
            out.append({**message, "tool_calls": decoded})
        return out

    # ----------------------------------------------------- schema helpers

    @staticmethod
    def as_output_type(schema: Dict[str, Any]):
        """Turn a raw JSON schema dict into something Outlines can constrain on.

        [VERIFY 2] Outlines v1 exposes `outlines.json_schema(...)`; some builds
        expose `outlines.types.JsonSchema(...)`. Both are tried here.

        Note: clients pass Pydantic models as MyModel.model_json_schema(), so
        the server never needs to import the client's Pydantic classes. That
        keeps the server generic -- it only ever speaks JSON Schema.
        """
        try:
            return outlines.json_schema(schema)
        except AttributeError:
            from outlines.types import JsonSchema
            return JsonSchema(schema)

    @staticmethod
    def tools_to_union_schema(tools: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build one schema that matches a call to any of the supplied tools.

        This is the safety net. Constraining to this union means the model
        physically cannot name a function you didn't define, or invent a
        parameter outside that function's schema. It can still pick the wrong
        tool or make up a plausible-but-wrong argument *value* -- grammars
        constrain shape, not truth.
        """
        branches = []
        for t in tools:
            fn = t.get("function", t)
            branches.append({
                "type": "object",
                "properties": {
                    # enum-of-one is more widely supported than const across
                    # schema->grammar compilers
                    "name": {"type": "string", "enum": [fn["name"]]},
                    "arguments": fn.get("parameters", {"type": "object"}),
                },
                "required": ["name", "arguments"],
                "additionalProperties": False,
            })
        return branches[0] if len(branches) == 1 else {"anyOf": branches}

    # --------------------------------------------------------- generation

    @staticmethod
    def sampler_for(temperature: Optional[float]):
        params = dict(CHAT_SAMPLING)
        if temperature is not None:
            params["temp"] = temperature
        return make_sampler(**params)

    @staticmethod
    def usage_block(prompt_tokens: int, completion_tokens: int) -> Dict[str, int]:
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

    def constrained_usage(self, prompt: str, text: str) -> Dict[str, int]:
        """Token counts for the constrained path.

        Outlines hands back only the finished string, so these are recovered by
        re-encoding instead of read off the decoder. Fine for accounting; do
        not treat them as exact the way the free path's counts are.
        """
        return self.usage_block(len(self.tokenizer.encode(prompt)),
                                len(self.tokenizer.encode(text)))

    def generate_free(self, prompt: str, max_tokens: int,
                      temperature: Optional[float] = None
                      ) -> Tuple[str, Dict[str, int]]:
        """Normal, unconstrained generation. Returns (text, usage).

        Built on stream_generate rather than generate because each
        GenerationResponse carries exact prompt_tokens/generation_tokens
        counts. Detokenizing and re-encoding to count them instead is slower
        and not always round-trip accurate.
        """
        pieces: List[str] = []
        prompt_tokens = completion_tokens = 0
        with self.gpu_lock:
            for r in mlx_lm.stream_generate(
                self.model, self.tokenizer, prompt,
                max_tokens=max_tokens,
                sampler=self.sampler_for(temperature),
                max_kv_size=self.max_kv_size,
            ):
                pieces.append(r.text)
                prompt_tokens = r.prompt_tokens
                completion_tokens = r.generation_tokens
        return "".join(pieces), self.usage_block(prompt_tokens, completion_tokens)

    def generate_constrained(self, prompt: str, schema: Dict[str, Any],
                             max_tokens: int) -> str:
        """Grammar-constrained generation. Output is guaranteed to match `schema`."""
        output_type = self.as_output_type(schema)
        with self.gpu_lock:
            # Outlines passes **kwargs straight through to mlx_lm.generate,
            # which forwards them to generate_step -- and generate_step takes
            # `sampler`, not `temp`. Splatting STRUCTURED_SAMPLING here raises
            # TypeError.
            return self.constrained_model(
                prompt,
                output_type,
                max_tokens=max_tokens,
                sampler=make_sampler(**STRUCTURED_SAMPLING),
                max_kv_size=self.max_kv_size,
            )

    # ------------------------------------------------------ wire envelopes

    def envelope(self,
                 content: Optional[str] = None,
                 tool_calls: Optional[List[Dict[str, Any]]] = None,
                 usage: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
        message: Dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls:
            message["tool_calls"] = tool_calls
        body = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model_id,
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": "tool_calls" if tool_calls else "stop",
            }],
        }
        if usage is not None:
            body["usage"] = usage
        return body

    def chunk(self, cid: str, created: int, *,
              delta: Optional[Dict[str, Any]] = None,
              finish_reason: Optional[str] = None,
              usage: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
        """One `chat.completion.chunk`.

        The trailing usage chunk that stream_options.include_usage asks for is
        the one case with an empty `choices` list -- that emptiness is how
        clients tell it apart from a content delta.
        """
        body: Dict[str, Any] = {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": created,
            "model": self.model_id,
            "choices": [],
        }
        if delta is not None or finish_reason is not None:
            body["choices"] = [{
                "index": 0,
                "delta": delta or {},
                "finish_reason": finish_reason,
            }]
        if usage is not None:
            body["usage"] = usage
        return body

    @staticmethod
    def sse(body: Dict[str, Any]) -> str:
        return f"data: {json.dumps(body)}\n\n"

    async def sse_chat(self, prompt: str, req: ChatRequest) -> AsyncIterator[str]:
        """Stream the plain-chat path as server-sent events.

        This is an *async* generator deliberately. Starlette drains a sync
        generator in a worker thread, which would walk straight back into the
        thread-local MLX stream crash; an async generator is driven by the
        event loop on the main thread, where the model and its gpu stream live.
        """
        cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())
        include_usage = bool((req.stream_options or {}).get("include_usage"))

        async with self.request_lock:
            yield self.sse(self.chunk(cid, created,
                                      delta={"role": "assistant", "content": ""}))

            prompt_tokens = completion_tokens = 0
            finish_reason = "stop"
            for r in mlx_lm.stream_generate(
                self.model, self.tokenizer, prompt,
                max_tokens=req.max_tokens,
                sampler=self.sampler_for(req.temperature),
                max_kv_size=self.max_kv_size,
            ):
                prompt_tokens = r.prompt_tokens
                completion_tokens = r.generation_tokens
                if r.finish_reason:
                    finish_reason = r.finish_reason
                if r.text:
                    yield self.sse(self.chunk(cid, created,
                                              delta={"content": r.text}))

            yield self.sse(self.chunk(cid, created, delta={},
                                      finish_reason=finish_reason))
            if include_usage:
                yield self.sse(self.chunk(
                    cid, created,
                    usage=self.usage_block(prompt_tokens, completion_tokens)))

        yield "data: [DONE]\n\n"

    @staticmethod
    def to_openai_tool_calls(parsed: Any) -> List[Dict[str, Any]]:
        calls = parsed if isinstance(parsed, list) else [parsed]
        out = []
        for c in calls:
            args = c.get("arguments", {})
            out.append({
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": c["name"],
                    "arguments": args if isinstance(args, str) else json.dumps(args),
                },
            })
        return out

    @staticmethod
    def looks_like_tool_attempt(text: str) -> bool:
        """Cheap heuristic: did the model try to call something and fumble it?

        Tune this to whatever your Gemma 4 build actually emits -- print `raw`
        for a few failed requests and look. Being wrong here is cheap in one
        direction (a wasted retry) and annoying in the other (prose returned
        where a tool call was wanted), so err toward retrying.
        """
        probes = ("tool_call", "```tool", "call:", '"name"')
        return any(p in text for p in probes)

    # ----------------------------------------------------------- handlers

    async def chat_completions(self, req: ChatRequest):
        # `async def` is load-bearing, not decoration. FastAPI runs a plain
        # `def` endpoint in an anyio worker thread, but MLX streams are
        # thread-local: the model and its gpu stream were created on the main
        # thread in __init__, so generating from a worker thread dies with
        #     RuntimeError: There is no Stream(gpu, 1) in current thread.
        # An async endpoint runs on the event loop -- i.e. the main thread --
        # which is the thread that owns the stream. Generation blocks that loop
        # for its duration, which is exactly what we want anyway: one GPU, one
        # request at a time (see gpu_lock and workers=1).

        # ---- Path 0: streaming ----------------------------------------------
        # Only the free path streams. A grammar-constrained decode hands back
        # one finished, schema-checked blob; there is no honest way to emit it
        # as incremental deltas, so say so instead of faking a stream.
        if req.stream:
            if req.tools or req.response_format:
                raise HTTPException(
                    status_code=400,
                    detail="stream=true is not supported with tools or "
                           "response_format on this server: constrained "
                           "decoding produces a single grammar-checked "
                           "result, not deltas.",
                )
            prompt = self.build_prompt(req.messages, thinking=req.thinking)
            # The lock is taken inside the generator, not here -- holding it
            # across the return would deadlock against the generator that
            # needs it.
            return StreamingResponse(self.sse_chat(prompt, req),
                                     media_type="text/event-stream")

        async with self.request_lock:
            # ---- Path 1: explicit schema wins over everything -----------------
            if req.response_format and req.response_format.get("type") == "json_schema":
                schema = req.response_format["json_schema"].get(
                    "schema", req.response_format["json_schema"]
                )
                prompt = self.build_prompt(req.messages, thinking=False)
                text = self.generate_constrained(prompt, schema, req.max_tokens)
                return self.envelope(content=text,
                                     usage=self.constrained_usage(prompt, text))

            # ---- Path 2: tools -> native first, constrained as fallback -------
            if req.tools:
                prompt = self.build_prompt(req.messages, tools=req.tools,
                                           thinking=req.thinking)
                raw, usage = self.generate_free(prompt, req.max_tokens,
                                                req.temperature)

                if gemma4_parser is not None:
                    try:
                        parsed = gemma4_parser.parse_tool_call(raw, req.tools)
                        if parsed:
                            return self.envelope(
                                tool_calls=self.to_openai_tool_calls(parsed),
                                usage=usage,
                            )
                    except Exception:
                        pass  # fall through to the constrained retry

                # The model either answered in prose or emitted something the
                # parser choked on. If it looks like an attempted call, force a
                # clean one.
                if self.looks_like_tool_attempt(raw):
                    union = self.tools_to_union_schema(req.tools)
                    fixed = self.generate_constrained(prompt, union,
                                                      req.max_tokens)
                    return self.envelope(
                        tool_calls=self.to_openai_tool_calls(json.loads(fixed)),
                        usage=self.constrained_usage(prompt, fixed),
                    )

                return self.envelope(content=raw, usage=usage)

            # ---- Path 3: plain chat -------------------------------------------
            prompt = self.build_prompt(req.messages, thinking=req.thinking)
            text, usage = self.generate_free(prompt, req.max_tokens,
                                             req.temperature)
            return self.envelope(content=text, usage=usage)

    def list_models(self) -> Dict[str, Any]:
        return {"object": "list",
                "data": [{"id": self.model_id, "object": "model"}]}


if __name__ == "__main__":
    Server().run()


# ==========================================================================
# CLIENT USAGE
# ==========================================================================
#
# from openai import OpenAI
# from pydantic import BaseModel
#
# client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="unused")
#
# # --- Pydantic-constrained output ---
# class Invoice(BaseModel):
#     vendor: str
#     total: float
#     currency: str
#
# r = client.chat.completions.create(
#     model="gemma-4-e2b",
#     messages=[{"role": "user", "content": "Acme Ltd, 249.99 EUR"}],
#     response_format={
#         "type": "json_schema",
#         "json_schema": {"name": "invoice",
#                         "schema": Invoice.model_json_schema()},
#     },
# )
# invoice = Invoice.model_validate_json(r.choices[0].message.content)
#
# # --- tool calling ---
# r = client.chat.completions.create(
#     model="gemma-4-e2b",
#     messages=[{"role": "user", "content": "Weather in Frankfurt?"}],
#     tools=[{
#         "type": "function",
#         "function": {
#             "name": "get_weather",
#             "description": "Current weather for a city.",
#             "parameters": {
#                 "type": "object",
#                 "properties": {"city": {"type": "string"}},
#                 "required": ["city"],
#             },
#         },
#     }],
# )
# print(r.choices[0].message.tool_calls)
#
#
# ==========================================================================
# NOTES FOR AN M1 AIR
# ==========================================================================
#
# Memory. 4-bit E2B is roughly 1.5-2 GB of weights, so it fits even on an 8 GB
# machine. The thing that will actually OOM you is the KV cache, which is why
# MAX_KV_SIZE is set. If you see mlx-lm's "model that requires..." warning,
# raise the wired memory limit:
#     sudo sysctl iogpu.wired_limit_mb=N     # N < your total RAM in MB
#
# Thinking mode vs. grammars. Gemma 4 ships configurable reasoning. Forcing a
# strict JSON grammar over a model mid-reasoning is a known trouble spot in the
# MLX stack -- mlx-vlm 0.6.x had exactly this failure, where a thinking budget
# shorter than the model's natural reasoning length made constrained generation
# either 500 or run away to max_tokens with invalid JSON. Keep thinking off on
# the constrained path, which is what this server does. If you want reasoning
# AND structure, do two passes: think unconstrained, then extract with a schema.
#
# Tool parser bugs. mlx-lm 0.31.2 had an open bug where the gemma4 parser threw
# "No function provided." on valid-looking calls. 0.31.3+ is the floor here.
# The constrained fallback above exists partly so a parser regression degrades
# into a slow request rather than a 500.
#
# Grammars constrain shape, not correctness. A schema guarantees you can call
# .model_validate_json() without a try/except. It guarantees nothing about
# whether the values are right. A 2B model will still confidently fill a
# required field with a plausible invention rather than leave it out -- so
# prefer Optional fields over required ones where "unknown" is a real answer.
#
# Speed. Gemma 4 ships a separate small drafter for speculative decoding
# (gemma-4-e2b-it-assistant-bf16). Worth wiring in via mlx-lm's draft model
# support once the basics work -- but get correctness first.
#
# If you need image or audio input, this file will not help you: mlx-lm runs
# Gemma 4's text path only. Switch to mlx-vlm, which has its own server with
# response_format support, and re-read the thinking-mode note above.
