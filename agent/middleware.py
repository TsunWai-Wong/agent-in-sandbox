"""Middleware: the checkpoints the agent loop stops at, and what they may say.

The loop calls the middleware. The middleware read the RunState. The state
calls nobody. That direction is the point — every place a run can be blocked,
paused or redirected is a line you can find by reading the loop, rather than a
handler that fires from somewhere else.

Middleware are expected to be fast and to hold no run state of their own. What
they carry is configuration; what they remember is in the ledger. That is what
makes them stateless, testable with a hand-built RunState and no API key, and
still correct after a pause that outlived the process.

The seam list is closed. Middleware plug in; seams do not multiply.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .event_store import RunState


# ---------------------------------------------------------------------------
# What a middleware may say
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Continue:
    """Nothing to say — ask the next middleware."""


@dataclass(frozen=True)
class Deny:
    """Don't run this. The reason goes back to the model so it can try again
    differently, which is why a bare "denied" is a wasted turn."""

    reason: str

    def __str__(self) -> str:
        return self.reason


@dataclass(frozen=True)
class Replace:
    """Run it, but with these arguments instead."""

    arguments: dict[str, Any]

    def __str__(self) -> str:
        return str(self.arguments)


@dataclass(frozen=True)
class Pause:
    """Freeze the run and ask a human. `key` is what the answer will be bound
    to, so an approval given for one tool call is not spent on the next."""

    question: str
    key: str = ""

    def __str__(self) -> str:
        return self.question


@dataclass(frozen=True)
class Stop:
    """End the run now."""

    reason: str

    def __str__(self) -> str:
        return self.reason


@dataclass(frozen=True)
class Inject:
    """Add this to what the model reads next.

    System-chosen middleware are invisible to the model — it has no idea a
    reviewer exists, it just sees a correction appear. So this text has to
    stand on its own: "Blocked by review: X" works, "see above" does not.
    """

    message: str

    def __str__(self) -> str:
        return self.message


Decision = Continue | Deny | Replace | Pause | Stop | Inject

CONTINUE = Continue()


@dataclass(frozen=True)
class HumanDecision:
    """What a person answered when a Pause reached them."""

    approved: bool
    key: str = ""
    note: str = ""


# ---------------------------------------------------------------------------
# The seams
# ---------------------------------------------------------------------------

DECISION_SEAMS = (
    "on_run_start",
    "before_model",
    "before_tool",
    "on_model_stop",
    "on_resume",
    "on_run_end",
)

# after_tool is a pipeline, not a decision: redacting secrets and capping output
# size must BOTH apply. Run through consult(), whichever was registered first
# would silently disable the other.
TRANSFORM_SEAMS = ("after_tool",)

SEAMS = DECISION_SEAMS + TRANSFORM_SEAMS


class Middleware:
    """Base class. Every hook defaults to a no-op, so a middleware implements
    only the seams it cares about."""

    def on_run_start(self, state: RunState, question: str) -> Decision:
        return CONTINUE

    def before_model(self, state: RunState) -> Decision:
        return CONTINUE

    def before_tool(self, state: RunState, call) -> Decision:
        return CONTINUE

    def on_model_stop(self, state: RunState, answer: str) -> Decision:
        return CONTINUE

    def on_resume(self, state: RunState, human: HumanDecision) -> Decision:
        return CONTINUE

    def on_run_end(self, state: RunState, answer: str) -> Decision:
        """The run is genuinely finished — every gate passed, nothing sent it
        back. Distinct from on_model_stop, which also fires on the passes where
        a reviewer rejects the answer and the model tries again."""
        return CONTINUE

    def after_tool(self, state: RunState, output: str, call) -> str:
        """Transform one tool result. Returns the output, not a decision."""
        return output


@dataclass(frozen=True)
class Registration:
    name: str
    cost: int
    mw: Middleware


class MiddlewareRegistry:
    """The middleware stack, indexed by seam and ordered by cost.

    Holds configuration only, never anything about a run in flight — which is
    what lets one registry serve every concurrent session. Run state arrives as
    an argument to consult().
    """

    def __init__(self) -> None:
        self._by_seam: dict[str, list[Registration]] = {s: [] for s in SEAMS}
        self._names: set[str] = set()

    def register(
        self, mw: Middleware, cost: int = 0, name: str | None = None
    ) -> "MiddlewareRegistry":
        """Register a middleware under every seam it implements.

        Raises if it implements none, which is nearly always a misspelled hook:
        `before_tools` registers cleanly, never fires, and looks exactly like a
        working guardrail. Same principle as ToolRegistry.register validating a
        schema — fail here rather than silently inside the loop.

        Cost is an ordering hint: 0 = free, 1 = seconds, 2 = an API call. Cheap
        deterministic checks run first so you never pay a model to judge what a
        regex already rejected.
        """
        name = name or type(mw).__name__
        if name in self._names:
            raise ValueError(f"middleware {name!r} is already registered")

        implemented = {
            seam
            for seam in SEAMS
            if getattr(type(mw), seam, None) is not getattr(Middleware, seam)
        }
        if not implemented:
            typos = sorted(
                attr
                for attr in vars(type(mw))
                if attr.startswith(("on_", "before_", "after_")) and attr not in SEAMS
            )
            raise ValueError(
                f"{name} implements no seam"
                + (f" — did you mean one of {SEAMS}? found {typos}" if typos else "")
            )

        for seam in implemented:
            self._by_seam[seam].append(Registration(name, cost, mw))
            # Stable, so equal costs keep registration order. Sorted per seam,
            # because two middleware at different seams never compete.
            self._by_seam[seam].sort(key=lambda r: r.cost)
        self._names.add(name)
        return self

    def consult(self, seam: str, state: RunState, *args) -> Decision:
        """Ask each middleware in cost order; the first non-Continue wins.

        Short-circuits for the same reason the gates do: never pay for a later
        check once an earlier one has already blocked. Recorded with the name of
        the middleware that decided, because "what stopped this run" is the
        question you actually have at 2am — and the loop, not the middleware,
        does the recording so nobody can forget.
        """
        if seam not in DECISION_SEAMS:
            raise KeyError(f"{seam!r} is not a decision seam: {DECISION_SEAMS}")

        for reg in self._by_seam[seam]:
            decision = getattr(reg.mw, seam)(state, *args)
            if not isinstance(decision, Continue):
                state.append(
                    seam,
                    type(decision).__name__.lower(),
                    key=getattr(decision, "key", ""),
                    by=reg.name,
                    detail=str(decision),
                )
                return decision
        return CONTINUE

    def transform(self, seam: str, state: RunState, value: str, *args) -> str:
        """Fold every middleware over a value — no short-circuit."""
        if seam not in TRANSFORM_SEAMS:
            raise KeyError(f"{seam!r} is not a transform seam: {TRANSFORM_SEAMS}")

        for reg in self._by_seam[seam]:
            value = getattr(reg.mw, seam)(state, value, *args)
        return value

    def fork(self, exclude: frozenset[str] = frozenset()) -> "MiddlewareRegistry":
        """A stack for a sub-agent — a different stack, by design.

        Excluding ReviewGate here is what stops a reviewer sub-agent from
        spawning its own reviewer, forever. Invariants and budgets usually stay;
        judgments and human approval usually don't.
        """
        child = MiddlewareRegistry()
        seen: set[str] = set()
        for seam in SEAMS:
            for reg in self._by_seam[seam]:
                if reg.name in exclude or reg.name in seen:
                    continue
                child.register(reg.mw, reg.cost, reg.name)
                seen.add(reg.name)
        return child

    def describe(self) -> str:
        """The whole pipeline, per seam, in firing order.

        A registry makes middleware easy to add, and twenty middleware means
        twenty places a run can stop. Being able to print the pipeline beats
        reconstructing it from wiring code.
        """
        blocks = []
        for seam in SEAMS:
            regs = self._by_seam[seam]
            if regs:
                rows = "\n".join(f"  [{r.cost}] {r.name}" for r in regs)
                blocks.append(f"{seam}:\n{rows}")
        return "\n".join(blocks) or "(no middleware registered)"
