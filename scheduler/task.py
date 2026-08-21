"""One scheduled task: when it runs, what it does, and who hears about it."""

from dataclasses import dataclass, field
from typing import Callable

from agent.agent import Agent
from agent.conversation import DEFAULT_TOKEN_BUDGET

from .delivery import Channel
from .outcome import OutcomeSink
from .policy import DeliverAlways, Policy
from .trigger import Trigger


@dataclass(frozen=True)
class ScheduledTask:
    """A definition, not a session: cheap, immutable, and safe to share.

    build_agent is a factory rather than a prebuilt Agent, and that is the one
    load-bearing decision here. Agent itself is immutable and shareable, but
    three things it is usually built with are not:

      * SkillRegistry._active mutates when load_skill lands, and the assembler
        rebuilds the instruction from it on every request — so a skill loaded by
        the digest run would appear in the monitor's instruction mid-loop. Call
        SkillRegistry.fork() here: the catalog stays shared, the loaded set does
        not, and no run re-reads the skills directory.
      * Browser wraps a single Playwright page, which two runs cannot share.
      * DockerSandbox writes every run into one workspace directory.

    Building them per fire is what makes a run isolated in the sense the guide
    means: fresh context, nothing inherited, nothing left behind.
    """

    id: str
    trigger: Trigger
    prompt: str
    build_agent: Callable[[OutcomeSink], Agent]
    channel: Channel
    policy: Policy = field(default_factory=DeliverAlways)
    token_budget: int = DEFAULT_TOKEN_BUDGET
    # Wall clock, unlike the agent loop's max_turns: ten turns of a model that
    # has stopped responding is still forever.
    deadline_seconds: int = 300
