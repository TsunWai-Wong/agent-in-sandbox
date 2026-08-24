import json
import logging
from typing import TYPE_CHECKING

from openinference.semconv.trace import SpanAttributes

from monitoring import get_tracer

from .context_assembler import ContextAssembler
from .event_store import AGENT_MESSAGE, TOOL_RESULT, USER_MESSAGE, RunState
from .llm_service import LLMService
from .middleware import (
    Deny,
    HumanDecision,
    Inject,
    MiddlewareRegistry,
    Pause,
    Replace,
    Stop,
)
from .providers import ChatResponse, Message, ToolResult
from .skills import SkillRegistry
from .tool_registry import ToolRegistry

if TYPE_CHECKING:
    # Type-only, and for UserMemoryStore that matters: memory/ imports agent/ at
    # runtime, so a real import here would put a cycle one careless line away.
    # The .search() call below is duck-typed.
    from memory.memory_store import UserMemoryStore

    from .session import Session

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)


class AgentLoopError(Exception):
    pass


class RunPaused(Exception):
    """A middleware asked for a human before the run may go on.

    Carries everything needed to resume — possibly in another process, hours
    later — which is the whole reason middleware return decisions instead of
    calling back. No thread is held while the answer is outstanding.
    """

    def __init__(self, state: RunState, pending: ChatResponse | None, decision: Pause):
        super().__init__(decision.question)
        # The state carries the event store, so the history rides along with it.
        self.state = state
        # The model turn whose tools have not run yet, or None when the pause
        # happened where nothing was waiting.
        self.pending = pending
        self.question = decision.question
        self.key = decision.key


class _Halted(Exception):
    """Internal: a Stop decision, travelling the way a Pause does.

    Stop and Pause both mean "this run does not carry on". Making them
    symmetric is what lets a seam be one line. Never escapes run_loop.
    """

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class Agent:
    def __init__(
        self,
        tools: ToolRegistry,
        llm: LLMService,
        instruction: str,
        skills: SkillRegistry | None = None,
        middleware: MiddlewareRegistry | None = None,
        memories: "UserMemoryStore | None" = None,
        user_id: str = "",
    ):
        self.tools = tools
        self.llm = llm
        # The instruction describes the agent, not a conversation, which is what
        # lets one agent be shared while histories stay per session.
        self.context = ContextAssembler(instruction)
        self.skills = skills
        self.middleware = middleware or MiddlewareRegistry()
        self.memories = memories
        # Whose memories. Not on RunState — a run does not own a user, a session
        # does. Empty disables recall as surely as memories=None.
        self.user_id = user_id

    # -- entry points ---------------------------------------------------------

    def run_loop(
        self,
        question: str | None = None,
        *,
        session: "Session | None" = None,
        state: RunState | None = None,
        max_turns: int = 10,
        files: list[str] | None = None,
        pending: ChatResponse | None = None,
    ) -> str:
        """Run until the model produces a final text response.

        Three ways to say which run this is, in precedence order:

            state     an existing run — resuming, or a sub-agent's own
            session   a new run in this conversation
            neither   an ephemeral run, owned by nobody

        Returns the answer alone. Everything else a caller might want — the
        history, the token spend — is on the state's event store, where it
        cannot be handed over stale.

        Raises RunPaused when a middleware wants a human; pass it to resume().
        """
        state = state or (
            session.begin_run(question or "")
            if session
            else RunState.begin(question or "")
        )
        with tracer.start_as_current_span(
            "run_loop", openinference_span_kind="agent"
        ) as span:
            # The question alone, never the attachments: a base64 payload on a
            # span attribute is megabytes of noise in the trace viewer.
            span.set_input(question or state.task)
            try:
                return self._drive(state, question, max_turns, span, files, pending)
            except _Halted as halted:
                return self._finish(span, state, halted.reason)

    def resume(
        self, paused: RunPaused, approved: bool, note: str = "", max_turns: int = 10
    ) -> str:
        """Answer a Pause and carry on.

        The answer is stamped against the key the pause carried, so it applies
        to that exact tool call and not to the next one sharing a name.
        Middleware re-run on the way through and find the stamp waiting.
        """
        state = paused.state
        state.append(
            "on_resume",
            "human_approved" if approved else "human_denied",
            key=paused.key,
            note=note,
        )
        self.middleware.consult(
            "on_resume", state, HumanDecision(approved, paused.key, note)
        )
        return self.run_loop(
            None, state=state, max_turns=max_turns, pending=paused.pending
        )

    # -- the loop -------------------------------------------------------------

    def _drive(
        self,
        state: RunState,
        question: str | None,
        max_turns: int,
        span,
        files: list[str] | None,
        pending: ChatResponse | None,
    ) -> str:
        """The choreography: which seam fires, and when.

        Deliberately the only place that names a seam. The mechanics live in the
        helpers below so that which checkpoints exist, and in what order, stays
        readable straight down this method.
        """
        if question is not None:
            state.record_message(USER_MESSAGE, self.llm.user_message(question, files))
            # Gates see the question as text. Input guardrails match on what was
            # written, and a parts list would silently stop every one of them.
            self._gate("on_run_start", state, question)

        tool_schemas = self.tools.get_schemas()

        for _ in range(max_turns):
            if pending is not None:
                # Resuming into a model turn whose tools never ran.
                response, pending = pending, None
            else:
                self._gate("before_model", state)
                response = self._call_model(state, tool_schemas, span)

            if response.tool_calls:
                self._run_tools(state, response)
                continue

            answer = response.text or ""
            state.record_message(AGENT_MESSAGE, Message(role="assistant", text=answer))

            if self._gate("on_model_stop", state, answer):
                # A gate sent the answer back. The model reads the correction as
                # an ordinary user turn, with no idea a reviewer exists.
                continue

            return self._finish(span, state, answer)

        raise AgentLoopError(f"Agent did not complete within {max_turns} turns")

    def _gate(self, seam: str, state: RunState, *args) -> bool:
        """Fire one whole-run seam and act on the answer.

        Continue and Inject are in-band. Stop and Pause are out-of-band, so both
        leave by exception rather than as a value a caller could forget to check.
        Returns whether something was injected.
        """
        decision = self.middleware.consult(seam, state, *args)
        if isinstance(decision, (Stop, Deny)):
            raise _Halted(str(decision))
        if isinstance(decision, Pause):
            # No pending here: every gate fires where a model turn is either
            # already consumed or never started.
            raise RunPaused(state, None, decision)
        if isinstance(decision, Inject):
            state.record_message(
                USER_MESSAGE, self.llm.user_message(decision.message)
            )
            return True
        return False

    def _call_model(self, state: RunState, tool_schemas: list, span) -> ChatResponse:
        """One model call, with the request rebuilt and the cost recorded."""
        # Reassembled every call, not once before the loop: this is what makes
        # load_skill take effect on the very next request rather than after the
        # answer is already written. Same for the history, which a mid-run
        # compaction may have just shortened.
        response = self.llm.chat(
            messages=self.context.build_messages(state.event_store),
            system=self.context.build_system_prompt(
                skills=self.skills.get_menu() if self.skills else None,
                skill_docs=self.skills.get_active_docs() if self.skills else None,
                memories=self._about_user(state.task),
            ),
            tools=tool_schemas,
        )
        usage = response.usage
        # One line per call, so attempts and spend are derived from the log
        # rather than counted. input_tokens separately, because it is the size of
        # what the model was asked to read — the quantity ContextBudget acts on.
        state.append(
            "before_model",
            "model_call",
            tokens=usage.total_tokens if usage else 0,
            input_tokens=usage.input_tokens if usage else 0,
            output_tokens=usage.output_tokens if usage else 0,
        )
        self._record_tokens(span, state)
        return response

    def _about_user(self, question: str) -> str | None:
        """The recall block, or None when there is nothing to say.

        Retrieval is an upgrade to an answer, never a precondition for one, and
        this runs inside every model call — so an unguarded query would turn a
        database hiccup into a failed reply.
        """
        if not (self.memories and self.user_id):
            return None
        try:
            rows = self.memories.search(self.user_id, question, with_directives=True)
        except Exception:
            logger.exception("memory unavailable; answering without it")
            return None
        return "\n".join(f"- {row['content']}" for row in rows) or None

    # -- tools ----------------------------------------------------------------

    def _run_tools(self, state: RunState, response: ChatResponse) -> None:
        """Consult on a whole batch of tool calls, then run it.

        Every call is consulted before any of it runs: a pause halfway would
        leave calls with no results, which every provider rejects on the next
        request. So the batch is all-or-nothing and a resume replays it whole.

        before_tool cannot use _gate — Deny and Replace answer about one call,
        not the run, so they are collected rather than raised.
        """
        decisions = []
        for call in response.tool_calls:
            decision = self.middleware.consult("before_tool", state, call)
            if isinstance(decision, Pause):
                raise RunPaused(state, response, decision)
            if isinstance(decision, Stop):
                raise _Halted(str(decision))
            decisions.append(decision)

        # The assistant turn carrying the calls goes down first, then one event
        # per result: a provider pairs them by call_id, and the store preserves
        # the order they were written in.
        state.record_message(
            AGENT_MESSAGE,
            Message(
                role="assistant",
                text=response.text or "",
                tool_calls=response.tool_calls,
            ),
        )
        for call, decision in zip(response.tool_calls, decisions):
            result = self._run_one(state, call, decision)
            state.record_message(
                TOOL_RESULT,
                Message(role="tool", text=result.output, tool_result=result),
                key=call.name,
            )

    def _run_one(self, state: RunState, call, decision) -> ToolResult:
        """Run one tool call, or report why it was refused."""
        with tracer.start_as_current_span(
            call.name, openinference_span_kind="tool"
        ) as span:
            span.set_attribute(SpanAttributes.TOOL_NAME, call.name)
            span.set_input(json.dumps(call.arguments))

            if isinstance(decision, Deny):
                # Fed back as an ordinary tool result so the history stays valid
                # and the model can pick another route.
                output = f"Refused: {decision.reason}"
            else:
                arguments = (
                    decision.arguments
                    if isinstance(decision, Replace)
                    else call.arguments
                )
                output = self.tools.dispatch(call.name, arguments, state)

            output = self.middleware.transform("after_tool", state, output, call)
            span.set_output(output)

        return ToolResult(call_id=call.id, name=call.name, output=output)

    # -- closing --------------------------------------------------------------

    def _finish(self, span, state: RunState, answer: str) -> str:
        """Fire on_run_end once, stamp the run, return.

        Distinct from on_model_stop, which also fires on passes where a gate
        sends the answer back. Any decision returned here is ignored — there is
        nothing left to redirect.
        """
        self.middleware.consult("on_run_end", state, answer)
        state.append("run", "run_finished", key=state.run_id, tokens=state.total_tokens)
        span.set_output(answer)
        return answer

    @staticmethod
    def _record_tokens(span, state: RunState) -> None:
        """Roll this run's token counts up onto the agent span.

        Summed off the log rather than counted alongside it. The
        auto-instrumentation already records each call; what it cannot show is
        what one question cost in total.
        """
        store, run = state.event_store, state.run_id
        prompt = store.sum("model_call", "input_tokens", run)
        completion = store.sum("model_call", "output_tokens", run)
        span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_PROMPT, prompt)
        span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION, completion)
        span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_TOTAL, prompt + completion)
