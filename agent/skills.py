import json
import logging
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

# Per document, not across the concatenation: trimming the joined block would
# cut whichever skill happened to be last in half. Same principle as
# MAX_LYRICS_CHARS — a section too big is trimmed where it is produced.
MAX_DOC_CHARS = 4_000


class Skill(BaseModel):
    """One skill directory, as read from disk.

    Frozen for the same reason ToolCall and Usage are: it is a value read once
    at scan time, and nothing downstream has any business editing it.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    doc: str
    # From the optional schema.json. Left as plain dicts rather than ToolSchema
    # because nothing registers them yet — see the note in load_skill.
    tools: list[dict] = Field(default_factory=list)


class SkillRegistry:
    """Registry with on-demand skill loading."""

    def __init__(self, skills_dir: str | Path):
        self.skills_dir = Path(skills_dir)
        self._catalog: dict[str, Skill] = {}
        self._active: dict[str, Skill] = {}
        self._scan()

    def _scan(self):
        """Scan the skills directory and build the catalog."""
        if not self.skills_dir.is_dir():
            # An absent directory means no skills yet, not a broken install:
            # this runs at startup, so raising here would take the app down
            # before a single skill had been written.
            return

        # Sorted, so the menu the model reads is in the same order every run.
        for skill_dir in sorted(self.skills_dir.iterdir()):
            if not skill_dir.is_dir():
                continue
            skill_md = skill_dir / "SKILL.md"
            schema_file = skill_dir / "schema.json"
            if not skill_md.exists():
                continue

            doc = skill_md.read_text()
            # Extract first line after "# " as description
            first_heading = ""
            for line in doc.splitlines():
                if line.startswith("# "):
                    first_heading = line[2:].strip()
                    break

            schemas = []
            if schema_file.exists():
                schemas = json.loads(schema_file.read_text())

            self._catalog[skill_dir.name] = Skill(
                name=skill_dir.name,
                description=first_heading,
                doc=doc,
                tools=schemas,
            )

    def get_menu(self) -> str:
        """Generate the skill menu for the model.

        Empty when nothing is installed, so ContextAssembler drops the section
        rather than advertising a menu with nothing on it. How to use the menu
        is the load_skill tool description's job, not this string's.

        The [loaded] marker cannot go stale: it and the docs section are both
        rendered from _active on the same request, so the menu never claims a
        skill the model can no longer read.
        """
        return "\n".join(
            f"- {name}: {skill.description}"
            + (" [loaded]" if name in self._active else "")
            for name, skill in self._catalog.items()
        )

    def get_active_docs(self) -> str:
        """The full instructions for every loaded skill.

        Rebuilt from _active on every request, and that is the whole point: a
        doc returned as a tool result would sit in the history where compact()
        strips it from older turns, and the skill would silently stop applying
        mid-conversation. Rederived into the instruction instead, it survives
        compaction because it was never subject to it.
        """
        return "\n\n".join(
            f"## {name}\n{self._cap(name, skill.doc)}"
            for name, skill in self._active.items()
        )

    def load_skill(self, name: str) -> str:
        """Activate a skill before answering a question it covers.

        The skill menu in your instructions lists only each skill's summary.
        Activating it adds its full instructions to your own instructions,
        where they stay until unloaded — so call this once and then follow them.

        Args:
            name: Skill name, exactly as it appears in the skill menu.
        """
        if name not in self._catalog:
            return f"Error: Unknown skill '{name}'. Check the skill menu."
        if name in self._active:
            return f"Skill '{name}' is already loaded."

        self._active[name] = self._catalog[name]
        # The doc itself is deliberately not returned here — get_active_docs()
        # puts it in the instruction on the very next model call, since the
        # assembler is rebuilt every iteration of the agent loop. Returning it
        # as well would just pay for the same text twice.
        return (
            f"Loaded skill '{name}'. Its full instructions are now in the "
            f"[Active skills] section of your instructions; follow them."
        )

    def unload_skill(self, name: str) -> str:
        """Unload a skill to free up context space."""
        if name not in self._active:
            return f"Skill '{name}' is not loaded."
        del self._active[name]
        return f"Unloaded skill '{name}'."

    @staticmethod
    def _cap(name: str, doc: str) -> str:
        """Trim one skill doc, loudly.

        This block is immune to compaction, so an oversized skill raises the
        floor of every later request in the session — worth a warning rather
        than a silent trim.
        """
        if len(doc) <= MAX_DOC_CHARS:
            return doc
        logger.warning(
            "Skill '%s' doc truncated: %d -> %d chars",
            name,
            len(doc),
            MAX_DOC_CHARS,
        )
        return doc[:MAX_DOC_CHARS] + "\n…[truncated]"
