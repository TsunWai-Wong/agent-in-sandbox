import logging
import os

from dotenv import load_dotenv

from src.agent.prompts import Prompt
from src.agent.providers import Usage
from src.agent.rag_agent import RAGAgent


logger = logging.getLogger(__name__)

# A cost ceiling, not the model's context limit. Both default models have
# windows far larger than this, so budgeting to the window would mean never
# compacting until a question had already cost a fortune.
DEFAULT_TOKEN_BUDGET = 60_000

# Two tiers, because there is no way to learn the post-eviction size without
# spending another call — and spending a call to measure would defeat the point
# of saving them. Escalating on the pre-compaction count instead is free:
# crossing the first threshold evicts, crossing the second evicts and then
# summarizes what is left.
EVICT_THRESHOLD = 0.5
COMPRESS_THRESHOLD = 0.7

KEEP_LAST_TURNS = 3


class Conversation:
    """One session's history, and the policy that keeps it from growing forever.

    The mechanics of reshaping history are per-provider and live in the
    adapters. What lives here is when to apply them, which is the same decision
    whichever model is answering.
    """

    agent: RAGAgent
    history: list
    last_usage: Usage | None

    def __init__(
        self,
        agent: RAGAgent,
        token_budget: int = DEFAULT_TOKEN_BUDGET,
        evict_threshold: float = EVICT_THRESHOLD,
        compress_threshold: float = COMPRESS_THRESHOLD,
        keep_last_turns: int = KEEP_LAST_TURNS,
        summary_model: str | None = None,
    ):
        load_dotenv()
        self.agent = agent
        self.history = []
        self.last_usage = None
        self.token_budget = token_budget
        self.evict_threshold = evict_threshold
        self.compress_threshold = compress_threshold
        self.keep_last_turns = keep_last_turns
        # Left as None so the provider's own default applies — both defaults
        # are already the cheap tier, and a model name valid for one provider
        # is not valid for the other.
        self.summary_model = summary_model or os.getenv("LLM_SUMMARY_MODEL") or None

    def reset(self) -> None:
        """Drop the conversation history.

        The agent's instruction is not part of it, so a reset conversation still
        behaves the same way — it has only forgotten what was said.
        """
        self.history = []
        # Or the next question inherits a token count it did not cause, and
        # compacts a history that is already empty.
        self.last_usage = None

    def ask(self, question: str) -> str:
        answer, self.history, usage = self.agent.agentic_loop(
            question, history=self.history
        )
        self.last_usage = usage
        if usage:
            # After answering, never before: this is latency the user would
            # otherwise wait through for no visible progress.
            self._manage_context(usage.input_tokens)
        return answer

    def _manage_context(self, used_tokens: int) -> None:
        """Shrink the history when the last call read too much."""
        if used_tokens < self.token_budget * self.evict_threshold:
            return

        before = len(self.history)
        self.history = self.agent.llm.compact(self.history, self.keep_last_turns)

        if used_tokens >= self.token_budget * self.compress_threshold:
            self.history = self._compress(self.history)

        logger.info(
            "Context management at %d tokens: history %d -> %d items",
            used_tokens,
            before,
            len(self.history),
        )

    def _compress(self, history: list) -> list:
        """Replace the older turns with a summary of them.

        Runs on already-evicted history, so the summarizer reads questions and
        answers and never a line of lyrics — which is what keeps this call
        cheap enough to be worth making.
        """
        older, recent = self.agent.llm.split_turns(history, self.keep_last_turns)
        if not older:
            return history

        try:
            summary = self.agent.llm.chat(
                messages=[
                    self.agent.llm.user_message(
                        self.agent.llm.render_transcript(older)
                    )
                ],
                system=Prompt.get_summary_instruction(),
                model=self.summary_model,
            ).text
        except Exception:
            # A long history costs tokens; a lost one costs the conversation.
            # Failing to summarize is not a reason to drop the turns anyway.
            logger.exception(
                "Summarizing the conversation failed; keeping the history as is."
            )
            return history

        if not summary:
            return history

        # Injected through user_message so it lands in whatever shape the
        # provider expects, and so the history still opens on a user turn —
        # which Gemini requires. The prefix is what stops the model reading it
        # as something the user just said.
        return [
            self.agent.llm.user_message(f"[Conversation summary]\n{summary}"),
            *recent,
        ]
