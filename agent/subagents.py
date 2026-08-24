"""Sub-agents: fresh context, own tools and skills, own middleware stack.

A sub-agent is a capability. It takes a SubTask, does a job, returns a
SubResult. It has no opinion about whether anyone waits for it — that is the
caller's business, which is why the same reviewer is blocking at one seam and
the same memory writer is fire-and-forget at another.

Two ways in, and the difference is who decides it runs:

    the model decides   -> expose_as_tool()   research, code search
    the system decides  -> a middleware       reviewer, memory writer

A reviewer the model can choose is a reviewer the model can decline, so
guardrail sub-agents are never given a tool adapter.
"""

from __future__ import annotations

import logging
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Callable

from opentelemetry import context as otel_context
from pydantic import BaseModel, ConfigDict, Field

from monitoring import get_tracer

from .event_store import RunState

if TYPE_CHECKING:  # pragma: no cover - import cycle: Agent imports middleware
    from .agent import Agent

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)


class DepthExceeded(RuntimeError):
    """A sub-agent tried to spawn past its spec's depth cap.

    max_turns bounds one loop; this bounds the tree. Without it, a sub-agent
    holding the delegate tool can spawn sub-agents forever.
    """


class Mode(str, Enum):
    """When the caller cashes the ticket.

    One mechanism, three policies — the spawn is identical, only the moment you
    call .result() differs.
    """

    BLOCKING = "blocking"  # immediately: reviewer before the final answer
    DEFERRED = "deferred"  # at a later seam: research that runs alongside
    ASYNC = "async"  # never: writing long-term memory


class SubTask(BaseModel):
    """The brief. Structured because the commonest sub-agent failure in
    practice is not a bad model, it is an under-specified task.

    Fresh context is the point of a sub-agent, and the price of it is that
    nothing the parent knows carries over unless it is written down here.
    """

    model_config = ConfigDict(frozen=True)

    task: str
    context: str = ""
    inputs: dict[str, str] = Field(default_factory=dict)
    success_criteria: str = ""
    returns: str = ""

    def render(self) -> str:
        blocks = [self.task]
        if self.context:
            blocks.append(f"[Context]\n{self.context}")
        for name, value in self.inputs.items():
            blocks.append(f"[Input: {name}]\n{value}")
        if self.success_criteria:
            blocks.append(f"[Done when]\n{self.success_criteria}")
        if self.returns:
            blocks.append(f"[Report back]\n{self.returns}")
        return "\n\n".join(blocks)


class SubResult(BaseModel):
    """What comes back. Carries usage so the parent's budget is not blind to
    the work it delegated."""

    model_config = ConfigDict(frozen=True)

    name: str
    ok: bool
    text: str = ""
    tokens: int = 0
    turns: int = 0
    run_id: str = ""
    error: str = ""


@dataclass(frozen=True)
class SubAgentSpec:
    """Recipe on how to build one sub-agent, and the limits it runs under.

    `build` is a factory rather than an instance because fresh context is not
    free: each spawn needs its own skills.fork(), so that one run's load_skill
    never shows up in another's instructions.
    """

    name: str
    build: Callable[[], "Agent"]
    default_mode: Mode = Mode.BLOCKING
    max_turns: int = 8
    max_depth: int = 1
    # Read by the model when this sub-agent is exposed as a tool. Unused
    # otherwise, since a system-chosen sub-agent is invisible to the model.
    tool_description: str = ""


@dataclass
class Handle:
    """A ticket for a spawned sub-agent. Cash it whenever the caller needs to —
    or never."""

    name: str
    seq: int
    _future: Future

    def done(self) -> bool:
        return self._future.done()

    def result(self, timeout: float | None = None) -> SubResult:
        return self._future.result(timeout=timeout)


class SubAgentRegistry:
    """Sub-agents by name. Configuration only — the RunState holds the run.

    Both trigger paths land on spawn(), so the depth check, the stamps and the
    token rollup are written once instead of drifting apart.
    """

    def __init__(self, pool: ThreadPoolExecutor | None = None, max_workers: int = 4):
        self._specs: dict[str, SubAgentSpec] = {}
        self._pool = pool or ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="subagent"
        )
        self._background: list[Handle] = []

    def register(self, spec: SubAgentSpec) -> "SubAgentRegistry":
        self._specs[spec.name] = spec
        return self

    def unregister(self, name: str) -> None:
        self._specs.pop(name, None)

    def get(self, name: str) -> SubAgentSpec:
        if name not in self._specs:
            raise KeyError(f"unknown sub-agent {name!r}; have {sorted(self._specs)}")
        return self._specs[name]

    def describe(self) -> str:
        return "\n".join(
            f"{s.name} [{s.default_mode.value}] turns={s.max_turns} depth<={s.max_depth}"
            for s in self._specs.values()
        ) or "(no sub-agents registered)"

    def spawn(
        self,
        name: str,
        task: SubTask,
        state: RunState,
        mode: Mode | None = None,
    ) -> Handle:
        """Dispatch a sub-agent and hand back a ticket.

        Mode defaults to the spec's, but the caller wins: the same sub-agent is
        waited for at one seam and thrown away at another, and that is the whole
        reason timing is not baked into the sub-agent itself.
        """
        spec = self.get(name)
        mode = mode or spec.default_mode

        if state.depth >= spec.max_depth:
            raise DepthExceeded(
                f"{name} refused at depth {state.depth} (max_depth={spec.max_depth})"
            )

        # Stamped before it runs, so a crash mid-flight still leaves a record of
        # what was attempted. Async work is invisible otherwise.
        dispatched = state.append(
            "spawn", "subagent_dispatched", key=name, mode=mode.value
        )
        child = state.child(task.task)

        # Captured here and attached inside the worker: OpenTelemetry context is
        # thread-local, so without this a pooled sub-agent's spans would show up
        # as orphan traces instead of under the run that paid for them.
        parent_context = otel_context.get_current()

        def job() -> SubResult:
            token = otel_context.attach(parent_context)
            try:
                with tracer.start_as_current_span(
                    f"subagent.{name}", openinference_span_kind="agent"
                ) as span:
                    span.set_input(task.render())
                    try:
                        agent = spec.build()
                        text = agent.run_loop(
                            task.render(), state=child, max_turns=spec.max_turns
                        )
                        span.set_output(text or "")
                        state.append(
                            "spawn",
                            "subagent_returned",
                            key=name,
                            of=dispatched.seq,
                            # Not run_id=: that is an Event field now, and the
                            # event belongs to the parent's run either way.
                            child_run_id=child.run_id,
                            tokens=child.total_tokens,
                        )
                        return SubResult(
                            name=name,
                            ok=True,
                            text=text or "",
                            tokens=child.total_tokens,
                            turns=child.attempts,
                            run_id=child.run_id,
                        )
                    except Exception as error:
                        # Fire-and-forget work fails where nobody is listening,
                        # so the ledger is the only place it can surface.
                        logger.exception("sub-agent %s failed", name)
                        state.append(
                            "spawn",
                            "subagent_failed",
                            key=name,
                            of=dispatched.seq,
                            child_run_id=child.run_id,
                            tokens=child.total_tokens,
                            error=str(error),
                        )
                        return SubResult(
                            name=name,
                            ok=False,
                            tokens=child.total_tokens,
                            run_id=child.run_id,
                            error=f"{type(error).__name__}: {error}",
                        )
            finally:
                otel_context.detach(token)

        if mode is Mode.BLOCKING:
            # Run inline rather than hopping to the pool: the caller is going to
            # block on it immediately anyway, and staying on this thread keeps
            # the span nesting natural and the pool free for real background work.
            future: Future = Future()
            future.set_result(job())
            return Handle(name, dispatched.seq, future)

        handle = Handle(name, dispatched.seq, self._pool.submit(job))
        if mode is Mode.ASYNC:
            # Nobody will ever call .result(), so the registry keeps the ticket:
            # a process that exits while the notebook is still being written
            # loses the write entirely.
            self._background.append(handle)
        return handle

    def join_background(self, timeout: float = 5.0) -> list[SubResult]:
        """Wait out any fire-and-forget work before the run walks out the door.

        "Async" means the answer did not block on it, not that it may be killed
        halfway. For work too slow to wait on at all, queue it durably instead
        of spawning it here.
        """
        results = []
        for handle in self._background:
            try:
                results.append(handle.result(timeout=timeout))
            except Exception as error:
                logger.warning("background sub-agent %s: %s", handle.name, error)
        self._background.clear()
        return results

    def expose_as_tool(self, tools, name: str) -> None:
        """Let the model delegate to a sub-agent by calling it.

        Only for sub-agents the model should CHOOSE. Never for a guardrail: one
        the model can call is one the model can decline.
        """
        spec = self.get(name)

        def handler(task: str, context: str = "", state: RunState = None) -> str:
            brief = SubTask(task=task, context=context)
            return self.spawn(name, brief, state, mode=Mode.BLOCKING).result().text

        # Built rather than written literally, so a spec can describe what its
        # specialist is for while the Args block — which is what tells the model
        # how to fill the parameters in — stays correct for every sub-agent.
        summary = spec.tool_description or (
            f"Delegate a self-contained sub-task to the {name} agent."
        )
        handler.__doc__ = (
            f"{summary}\n"
            "\n"
            "Args:\n"
            "    task: What to do, written so it stands alone — the specialist\n"
            "        starts fresh and sees none of this conversation.\n"
            "    context: Facts it needs that it cannot look up for itself.\n"
        )
        tools.register(f"delegate_{name}", handler, wants_state=True)
