import json
from collections.abc import Callable
from dataclasses import dataclass
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
from .providers import ChatResponse, Message, ToolResult, Usage
from .skills import SkillRegistry
from .tool_registry import ToolRegistry

if TYPE_CHECKING:
    # Type-only: Session imports nothing from here, and the agent must not
    # need a session to be constructed.
    from .session import Session

tracer = get_tracer(__name__)

class AgentLoopError(Exception):
    pass


class RunPaused(Exception):
    """A middleware asked for a human before the run may go on.

    Carries everything needed to pick the run up again — possibly in another
    process, possibly hours later — because that is the whole reason the ledger
    is a passive record rather than a set of live callbacks. Nothing is blocked
    and no thread is held while the answer is outstanding.
    """

    def __init__(self, state: RunState, pending, decision: Pause):
        super().__init__(decision.question)
        # The state carries the event store, so the history rides along with it
        # — there is nothing left for the caller to hold on to separately.
        self.state = state
        # The model turn whose tools have not run yet, or None when the pause
        # happened somewhere no tools were waiting.
        self.pending = pending
        self.question = decision.question
        self.key = decision.key


class _Halted(Exception):
    """Internal: a Stop decision, travelling the same way a Pause does.

    Stop and Pause are the only two verbs that mean "this run does not carry
    on". Having one return a value while the other raised forced every seam to
    handle both by hand; making them symmetric is what lets a seam be one line.
    Never escapes agentic_loop — it is caught and turned into an ordinary return.
    """

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass
class _Cursor:
    """Everything one run mutates as it goes.

    Bundled so the helpers below can take a single argument instead of five,
    and so the token bookkeeping lives next to the history it describes rather
    than as four parallel locals that have to be threaded through by hand.
    """

    state: RunState
    # No messages here any more: the history is the event store, and rebuilding
    # it per call is what lets a summarization appended mid-run take effect on
    # the very next request.
    # A model turn whose tools have not run yet, set only when resuming.
    pending: ChatResponse | None = None
    last_usage: Usage | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def record(self, usage: Usage | None) -> None:
        if not usage:
            return
        self.last_usage = usage
        self.prompt_tokens += usage.input_tokens
        self.completion_tokens += usage.output_tokens


class Agent:
    tools: ToolRegistry
    llm: LLMService
    context: ContextAssembler
    skills: SkillRegistry | None
    middleware: MiddlewareRegistry


    def __init__(
        self,
        tools: ToolRegistry,
        llm: LLMService,
        instruction: str,
        skills: SkillRegistry | None = None,
        middleware: MiddlewareRegistry | None = None,
        memories: Callable[[], str] | None = None,
    ):
        self.tools = tools
        self.llm = llm
        # The instruction describes the agent, not any one conversation, so it
        # sits alongside the tools and the model. Being immutable is what lets
        # an agent be built once and shared while histories stay per session.
        # The assembler owns it now: what the model reads is the instruction
        # plus whatever sections apply to this request.
        self.context = ContextAssembler(instruction)
        # Optional, so the eval scripts and any caller with no skills directory
        # keep working unchanged — the menu section simply never appears.
        self.skills = skills
        # Also optional: an empty registry consults nobody, so an agent built
        # the old way behaves exactly as it did before the seams existed.
        self.middleware = middleware or MiddlewareRegistry()
        # A callable rather than a store, for the same reason skills is a
        # registry: the agent needs the rendered block at call time and has no
        # business knowing where it came from. Leaving it None is the recall
        # half of the memory switch — nothing is fetched and no section appears.
        self.memories = memories

    def agentic_loop(
        self,
        question: str | None = None,
        max_turns: int = 10,
        state: RunState | None = None,
        pending: ChatResponse | None = None,
        files: list[str] | None = None,
    ) -> tuple[str, Usage | None]:
        """Run the agentic loop until the model produces a final text response.

        Returns the answer and the usage of the final call. No history: it is
        recorded on the state's event store as it happens, so the caller who
        owns the session already has it and cannot be handed a stale copy.

        The usage returned is the last call's, not the turn's total, because
        its input_tokens is the size of everything the model was asked to read
        — which is the quantity a caller managing context needs to act on. The
        turn's total is a different question and lives on the ledger, where a
        sub-agent's spend can be rolled into it.

        Raises RunPaused when a middleware wants a human. Pass the exception
        back to resume() once the answer is in.
        """
        state = state or RunState.begin(task=question or "")
        cursor = _Cursor(state=state, pending=pending)

        with tracer.start_as_current_span(
            "agentic_loop", openinference_span_kind="agent"
        ) as span:
            # The question alone, never the attachments: a base64 payload on a
            # span attribute is megabytes of noise in the trace viewer.
            span.set_input(question or state.task)
            try:
                return self._drive(cursor, question, max_turns, span, files)
            except _Halted as halted:
                return self._finish(span, cursor, halted.reason)

    def _drive(
        self,
        cursor: _Cursor,
        question: str | None,
        max_turns: int,
        span,
        files: list[str] | None = None,
    ) -> tuple[str, Usage | None]:
        """The choreography: which seam fires, and when.

        Deliberately the only place that names a seam, and deliberately short.
        The mechanics of applying a decision, calling the model and running a
        tool batch live in the helpers below — but which checkpoints exist and
        what order they come in stays readable straight down this method, since
        that is the property the whole design is for.
        """
        if question is not None:
            cursor.state.record_message(
                USER_MESSAGE, self.llm.user_message(question, files)
            )
            # The gate still sees the question as text. Input guardrails match
            # on what was written, and handing them a parts list would silently
            # stop every one of them from matching anything.
            self._gate("on_run_start", cursor, question)

        tool_schemas = self.tools.get_schemas()

        for _ in range(max_turns):
            if cursor.pending is not None:
                # Resuming into a model turn whose tools never ran. Nothing to
                # re-ask — the response is already in hand.
                response, cursor.pending = cursor.pending, None
            else:
                self._gate("before_model", cursor)
                response = self._call_model(cursor, tool_schemas, span)

            if response.tool_calls:
                self._run_tools(cursor, response)
                continue

            # Record the final turn too, or the next question in the
            # conversation would not see the answer it follows from.
            answer = response.text or ""
            cursor.state.record_message(
                AGENT_MESSAGE, Message(role="assistant", text=answer)
            )

            if self._gate("on_model_stop", cursor, answer):
                # A gate sent the answer back. The model reads the correction as
                # an ordinary user turn, having no idea a reviewer exists.
                continue

            return self._finish(span, cursor, answer)

        raise AgentLoopError(f"Agent did not complete within {max_turns} turns")

    # -- the mechanics ---------------------------------------------------------

    def _gate(self, seam: str, cursor: _Cursor, *args) -> bool:
        """Fire one whole-run seam and act on the answer.

        Two categories, and that is what collapses this to a single call site
        line. Continue and Inject are in-band: the loop carries on, with any
        injected text added to what the model reads next. Stop and Pause are
        out-of-band: the run is over or suspended, so both leave by exception
        rather than as a value a caller could forget to check.

        Returns whether something was injected, which is the only thing a caller
        still has a choice about.
        """
        decision = self.middleware.consult(seam, cursor.state, *args)
        if isinstance(decision, (Stop, Deny)):
            raise _Halted(str(decision))
        if isinstance(decision, Pause):
            raise RunPaused(cursor.state, cursor.pending, decision)
        if isinstance(decision, Inject):
            cursor.state.record_message(
                USER_MESSAGE, self.llm.user_message(decision.message)
            )
            return True
        return False

    def _call_model(self, cursor: _Cursor, tool_schemas: list, span) -> ChatResponse:
        """One model call, with the instruction rebuilt and the cost recorded."""
        # Reassembled every call, not once before the loop: this is what makes
        # load_skill take effect. The call lands mid-turn, and the doc it
        # activated has to be in the instruction on the very next request rather
        # than after the answer is already written.
        response = self.llm.chat(
            messages=self.context.build_messages(cursor.state.event_store),
            system=self.context.build_system_prompt(
                skills=self.skills.get_menu() if self.skills else None,
                skill_docs=self.skills.get_active_docs() if self.skills else None,
                memories=self.memories() if self.memories else None,
            ),
            tools=tool_schemas,
        )
        cursor.record(response.usage)
        # Rewritten every iteration rather than once at the end, so the rollup is
        # on the span whichever way the loop exits.
        self._record_tokens(span, cursor.prompt_tokens, cursor.completion_tokens)
        # One line per model call, so attempts and token spend are derived from
        # the ledger instead of counted by middleware.
        # input_tokens separately from the total, because it is the size of what
        # the model was asked to read — the quantity ContextBudget acts on.
        cursor.state.append(
            "before_model",
            "model_call",
            tokens=(response.usage.total_tokens if response.usage else 0),
            input_tokens=(response.usage.input_tokens if response.usage else 0),
            output_tokens=(response.usage.output_tokens if response.usage else 0),
        )
        return response

    def _run_tools(self, cursor: _Cursor, response: ChatResponse) -> None:
        """Consult on a whole batch of tool calls, then run it.

        Every call is consulted before any of it runs. A pause halfway would
        leave the history holding tool calls with no results, which every
        provider rejects on the next request — so the batch is all-or-nothing,
        and a resume replays it from the top.

        before_tool cannot use _gate: Deny and Replace are answers about one
        call, not about the run, so they are collected rather than raised.
        """
        decisions = []
        for call in response.tool_calls:
            decision = self.middleware.consult("before_tool", cursor.state, call)
            if isinstance(decision, Pause):
                raise RunPaused(cursor.state, response, decision)
            if isinstance(decision, Stop):
                raise _Halted(str(decision))
            decisions.append(decision)

        # The assistant turn carrying the calls goes down first, then one event
        # per result: a provider pairs them by call_id, and the store preserves
        # the order they were written in.
        cursor.state.record_message(
            AGENT_MESSAGE,
            Message(
                role="assistant", text=response.text or "", tool_calls=response.tool_calls
            ),
        )
        for call, decision in zip(response.tool_calls, decisions):
            result = self._run_one(cursor, call, decision)
            cursor.state.record_message(
                TOOL_RESULT,
                Message(role="tool", text=result.output, tool_result=result),
                key=call.name,
            )

    def _run_one(self, cursor: _Cursor, call, decision) -> ToolResult:
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
                output = self.tools.dispatch(call.name, arguments, cursor.state)

            output = self.middleware.transform("after_tool", cursor.state, output, call)
            span.set_output(output)

        return ToolResult(call_id=call.id, name=call.name, output=output)

    def _finish(self, span, cursor: _Cursor, answer: str):
        """Close a run: fire on_run_end once, stamp it, return.

        Distinct from on_model_stop, which also fires on the passes where a gate
        sends the answer back. Fire-and-forget work belongs here, where the run
        is genuinely over — and any decision it returns is ignored, because
        there is nothing left to redirect.
        """
        state = cursor.state
        self.middleware.consult("on_run_end", state, answer)
        state.append("run", "run_finished", key=state.run_id, tokens=state.total_tokens)
        span.set_output(answer)
        return answer, cursor.last_usage

    # -- asking ----------------------------------------------------------------

    def ask(
        self,
        session: "Session",
        question: str,
        files: list[str] | None = None,
        max_turns: int = 10,
    ) -> str:
        """Ask one question in a session. What Conversation.ask used to be.

        The agent acts; the session holds the state. That direction is why this
        lives here rather than on Session — an Agent already takes a RunState,
        and taking the session it belongs to is the same move one level up.

        Nothing to hand back and forth: the answer goes onto the session's event
        store as it is produced, so a caller cannot hold a stale history.
        """
        answer, _ = self.agentic_loop(
            question, state=session.begin_run(question), max_turns=max_turns, files=files
        )
        return answer

    # -- resuming --------------------------------------------------------------

    def resume(
        self,
        paused: RunPaused,
        approved: bool,
        note: str = "",
        max_turns: int = 10,
    ) -> tuple[str, Usage | None]:
        """Answer a Pause and carry on.

        The answer is stamped against the key the pause carried, so it applies
        to that exact tool call and not to the next one that merely shares a
        name. Middleware re-run on the way through and find the stamp waiting.
        """
        state = paused.state
        state.append(
            "on_resume",
            "human_approved" if approved else "human_denied",
            key=paused.key,
            note=note,
        )
        human = HumanDecision(approved=approved, key=paused.key, note=note)
        self.middleware.consult("on_resume", state, human)

        return self.agentic_loop(
            question=None,
            max_turns=max_turns,
            state=state,
            pending=paused.pending,
        )

    @staticmethod
    def _record_tokens(span, prompt_tokens: int, completion_tokens: int) -> None:
        """Roll the turn's token counts up onto the agent span.

        The auto-instrumentation already records each call separately; what it
        cannot show is what one user question cost in total, which is the
        number that says whether context management is working.
        """
        span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_PROMPT, prompt_tokens)
        span.set_attribute(
            SpanAttributes.LLM_TOKEN_COUNT_COMPLETION, completion_tokens
        )
        span.set_attribute(
            SpanAttributes.LLM_TOKEN_COUNT_TOTAL, prompt_tokens + completion_tokens
        )
