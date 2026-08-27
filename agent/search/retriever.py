"""The search the agent calls: rewrite the question, fan out, fuse, render.

Everything corpus-specific lives here rather than in the tool schema, so the
model supplies a question and nothing else.
"""

from __future__ import annotations

from ..event_store import RunState
from .hybrid_search import HybridSearch
from .query_rewriter import QueryRewriter, recent_turns


# An embedding is thousands of floats and means nothing to a model.
SKIP_FIELDS = frozenset({"embedding", "score"})


class Retriever:
    """Rewriting and hybrid search behind a single tool call.

    Args:
        rewriter: Turns the question into the queries to run.
        hybrid_searcher: Runs them and fuses the rankings.
        text_query: Elasticsearch query body, with "{query}" where the query
            text belongs.
        vector_query: Nearest-neighbour statement for the vector store.
        num_results: How many documents come back.
        fields: Which fields to show the model. All of them, minus the
            unreadable ones, when left unset.
    """

    def __init__(
        self,
        rewriter: QueryRewriter,
        hybrid_searcher: HybridSearch,
        text_query: dict,
        vector_query: str,
        num_results: int = 5,
        fields: list[str] | None = None,
    ) -> None:
        self.rewriter = rewriter
        self.hybrid_searcher = hybrid_searcher
        self.text_query = text_query
        self.vector_query = vector_query
        self.num_results = num_results
        self.fields = fields

    def search(self, query: str, state: RunState = None) -> str:
        """Search the indexed documents and return the closest matches.

        Args:
            query: What you are looking for, in plain language. Ask for
                everything you need at once — a question covering several
                things is split and searched for separately.
        """
        queries = self.rewriter.rewrite(query, recent_turns(state))
        documents = self.hybrid_searcher.search(
            queries, self.text_query, self.vector_query, self.num_results
        )
        return self._render(documents)

    def _render(self, documents: list[dict]) -> str:
        """Documents as text. Numbered, so an answer can cite one."""
        if not documents:
            return "No matching documents."

        blocks = []
        for position, document in enumerate(documents, 1):
            names = self.fields or [f for f in document if f not in SKIP_FIELDS]
            body = "\n".join(
                f"{name}: {document[name]}" for name in names if name in document
            )
            blocks.append(f"[{position}]\n{body}")
        return "\n\n".join(blocks)
