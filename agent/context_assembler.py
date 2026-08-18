import logging

logger = logging.getLogger(__name__)

# Caps, not a token budget: a section too big is trimmed where it is produced,
# so what reaches the model is predictable rather than whatever happened to fit.
# Same reasoning as MAX_LYRICS_CHARS in search_tools.
MAX_SKILLS_CHARS = 2_000
# A backstop only: SkillRegistry caps each doc as it renders it, which is the
# cap that matters, since trimming the joined block would halve whichever skill
# came last.
MAX_SKILL_DOCS_CHARS = 16_000
MAX_PINS_CHARS = 1_000


class ContextAssembler:
    """Assembles one request's instruction from the parts in play.

    Order is fixed and static-first. The agent instruction never changes, so
    keeping it at the front leaves a stable prefix for provider-side prompt
    caching and puts every volatile section behind it — the skill menu, which
    moves only when a skill loads, ahead of anything that changes per turn.

    Nothing here is ever stored in the conversation history — it is rebuilt on
    every request. That is the point of the split: history holds what cannot be
    rederived, this holds what can. An active skill's doc is rederivable from
    the registry, which is why it belongs here: a doc returned as a tool result
    would live in the history and compact() would strip it from older turns,
    so the skill would quietly stop applying part-way through a conversation.
    """

    instruction: str

    def __init__(self, instruction: str) -> None:
        self.instruction = instruction

    def build(
        self,
        skills: str | None = None,
        skill_docs: str | None = None,
        pins: str | None = None,
    ) -> str:
        blocks = [self.instruction]

        for name, content, cap in (
            ("Available skills", skills, MAX_SKILLS_CHARS),
            ("Active skills", skill_docs, MAX_SKILL_DOCS_CHARS),
            ("Songs under discussion", pins, MAX_PINS_CHARS),
        ):
            if not content:
                continue
            if len(content) > cap:
                # Logged, never silent: a section that vanished without a trace
                # looks like a model failure rather than a budget decision.
                logger.warning(
                    "%s truncated: %d -> %d chars", name, len(content), cap
                )
                content = content[:cap] + "\n…[truncated]"
            blocks.append(f"[{name}]\n{content}")

        return "\n\n".join(blocks)
