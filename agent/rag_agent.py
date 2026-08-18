import json

from openinference.semconv.trace import SpanAttributes

from src.monitoring import get_tracer

from .context_assembler import ContextAssembler
from .llm_service import LLMService
from .providers import ToolResult, Usage
from .skills import SkillRegistry
from .tool_registry import ToolRegistry

tracer = get_tracer(__name__)

class AgentLoopError(Exception):
    pass

class RAGAgent:
    tools: ToolRegistry
    llm: LLMService
    context: ContextAssembler
    skills: SkillRegistry | None


    def __init__(
        self,
        tools: ToolRegistry,
        llm: LLMService,
        instruction: str,
        skills: SkillRegistry | None = None,
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

    def agentic_loop(
        self,
        question: str,
        history: list | None = None,
        max_turns: int = 10,
    ) -> tuple[str, list, Usage | None]:
        """Run the agentic loop until the model produces a final text response.

        Returns the answer, the extended history, and the usage of the final
        call. That history is in whatever shape the active provider uses; it is
        only ever passed back in, never read here, which is what keeps this
        loop provider-agnostic.

        The usage returned is the last call's, not the turn's total, because
        its input_tokens is the size of everything the model was asked to read
        — which is the quantity a caller managing context needs to act on.
        """
        with tracer.start_as_current_span(
            "agentic_loop", openinference_span_kind="agent"
        ) as agent_span:
            agent_span.set_input(question)

            messages = [*(history or []), self.llm.user_message(question)]
            tool_schemas = self.tools.get_schemas()

            prompt_tokens = 0
            completion_tokens = 0
            last_usage = None

            for _ in range(max_turns):
                # Reassembled every iteration, not once before the loop: this is
                # what makes load_skill take effect. The call lands mid-turn, and
                # the doc it activated has to be in the instruction on the very
                # next request rather than after the answer is already written.
                response = self.llm.chat(
                    messages=messages,
                    system=self.context.build(
                        skills=self.skills.get_menu() if self.skills else None,
                        skill_docs=(
                            self.skills.get_active_docs() if self.skills else None
                        ),
                    ),
                    tools=tool_schemas,
                )

                if response.usage:
                    last_usage = response.usage
                    prompt_tokens += response.usage.input_tokens
                    completion_tokens += response.usage.output_tokens
                    # Rewritten every iteration rather than once at the end, so
                    # the rollup is on the span whichever way the loop exits.
                    self._record_tokens(
                        agent_span, prompt_tokens, completion_tokens
                    )

                if not response.tool_calls:
                    # Record the final turn too, or the next question in the
                    # conversation would not see the answer it follows from.
                    messages = self.llm.extend(messages, response, [])
                    agent_span.set_output(response.text or "")
                    return response.text, messages, last_usage

                results = []
                for call in response.tool_calls:
                    with tracer.start_as_current_span(
                        call.name, openinference_span_kind="tool"
                    ) as tool_span:
                        tool_span.set_attribute(SpanAttributes.TOOL_NAME, call.name)
                        tool_span.set_input(json.dumps(call.arguments))
                        output = self.tools.dispatch(call.name, call.arguments)
                        tool_span.set_output(output)
                    results.append(
                        ToolResult(call_id=call.id, name=call.name, output=output)
                    )

                messages = self.llm.extend(messages, response, results)

            raise AgentLoopError(f"Agent did not complete within {max_turns} turns")

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
