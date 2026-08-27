"""Query rewriting: the question as asked, turned into the queries to run.

One call does both jobs. A question that already stands alone comes back as a
single cleaned-up query; a compound one comes back as several. Callers get a
list either way and never have to ask which happened.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from ..event_store import AGENT_MESSAGE, USER_MESSAGE, RunState
from ..llm_service import LLMService
from ..prompts import Prompt


logger = logging.getLogger(__name__)

# Referents are local: "its lyrics" points at the last turn or two, never at
# turn three of twenty.
HISTORY_TURNS = 5

# Every extra query is one more search per backend.
MAX_QUERIES = 3

# Enough of a turn to resolve what the next one refers to.
MAX_TURN_CHARS = 500

SPEAKERS = {USER_MESSAGE: "User", AGENT_MESSAGE: "Assistant"}


class RewrittenQueries(BaseModel):
    queries: list[str] = Field(default_factory=list)


def recent_turns(state: RunState | None, limit: int = HISTORY_TURNS) -> str:
    """The last few spoken turns, for resolving what a question refers to.

    Both speakers, because the thing an ambiguous question points at is as often
    in the answer as in the question.

    Tool results are left out: they are large, and they are mostly documents
    already retrieved, which would steer the rewrite toward what has been found
    rather than what is missing.
    """
    if state is None:
        return ""

    lines = [
        f"{SPEAKERS[event.kind]}: {event.text[:MAX_TURN_CHARS]}"
        for event in state.session_history()
        if event.kind in SPEAKERS and event.text
    ]
    return "\n".join(lines[-limit:])


class QueryRewriter:
    """Turns one question into the queries worth searching for."""

    def __init__(self, llm: LLMService, max_queries: int = MAX_QUERIES) -> None:
        self.llm = llm
        self.max_queries = max_queries

    def rewrite(self, question: str, history: str = "") -> list[str]:
        """The queries to search for, best effort.

        Never raises and never returns empty. Anything that goes wrong falls
        back to the question as asked, which is a serviceable query on its own —
        a broken rewriter degrades retrieval instead of taking it down.
        """
        try:
            response = self.llm.chat(
                messages=[self.llm.user_message(_payload(question, history))],
                system=Prompt.get_rewrite_instruction(),
                text_format=RewrittenQueries,
            )
        except Exception:
            logger.exception("rewrite failed; searching the question as asked")
            return [question]

        if not isinstance(response.parsed, RewrittenQueries):
            return [question]

        queries = [q.strip() for q in response.parsed.queries if q and q.strip()]
        return queries[: self.max_queries] or [question]


def _payload(question: str, history: str) -> str:
    if not history:
        return question
    return f"[Conversation so far]\n{history}\n\n[Question]\n{question}"
