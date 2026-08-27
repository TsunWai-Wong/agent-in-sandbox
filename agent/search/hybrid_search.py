from monitoring import get_tracer, set_documents

from .text_search import TextSearch
from .vector_search import VectorSearch

tracer = get_tracer(__name__)

# What a text_query template gets the current query substituted into.
QUERY_PLACEHOLDER = "{query}"


def fill_query(node, query: str):
    """Put the query into a template body, wherever the placeholder sits.

    Walks the dict rather than formatting a string, so a query containing a
    quote or a brace cannot damage the body around it.
    """
    if isinstance(node, dict):
        return {key: fill_query(value, query) for key, value in node.items()}
    if isinstance(node, list):
        return [fill_query(value, query) for value in node]
    if isinstance(node, str):
        return node.replace(QUERY_PLACEHOLDER, query)
    return node

class HybridSearch():
    text_searcher: TextSearch
    vector_searcher: VectorSearch

    def __init__(self, text_searcher: TextSearch,
                 vector_searcher: VectorSearch) -> None:
        self.text_searcher = text_searcher
        self.vector_searcher = vector_searcher

    def _rrf(self, result_lists, k=60, num_results=5):
        """Fuse ranked lists, returning the top documents with their RRF scores."""
        scores = {}
        docs = {}

        for results in result_lists:
            for rank, doc in enumerate(results):
                key = (doc["id"])
                scores[key] = scores.get(key, 0) + 1 / (k + rank)
                docs[key] = doc

        ranked = sorted(scores, key=scores.get, reverse=True)
        return [(docs[key], scores[key]) for key in ranked[:num_results]]

    def search(
        self,
        queries: list[str],
        text_query: dict,
        vector_query: str,
        num_results: int = 10,
    ) -> list[dict]:
        """Search the corpus for every query and fuse the rankings into one.

        Each query runs against both Elasticsearch and the vector store, and all
        the rankings go into a single Reciprocal Rank Fusion. Fusing rather than
        concatenating is what makes decomposition worth it: a document several
        queries agree on outranks one that only the best query found.

        One query is the one-element case, so there is nothing else to call.

        Args:
            queries: The queries to run. Usually from a QueryRewriter.
            text_query: Elasticsearch query body, with "{query}" wherever the
                query text belongs — it is filled in per query, so each one
                searches for itself rather than all of them repeating the first.
                Its `size` is replaced with the pool size, so every ranking is
                fused over equal depth.
            vector_query: Nearest-neighbour statement for the vector store.
            num_results: Maximum number of documents to return.
        """
        with tracer.start_as_current_span(
            "hybrid_search", openinference_span_kind="chain"
        ) as span:
            span.set_input(queries)
            pool_size = max(2 * num_results, 10)
            span.set_attribute("retriever.pool_size", pool_size)
            span.set_attribute("retriever.num_queries", len(queries))

            rankings = []
            for query in queries:
                rankings.append(
                    self.text_searcher.text_search(
                        {**fill_query(text_query, query), "size": pool_size}
                    )
                )
                rankings.append(
                    self.vector_searcher.vector_search(
                        query, vector_query, num_results=pool_size
                    )
                )

            fused = self._rrf(rankings, num_results=num_results)
            documents = [document for document, _ in fused]
            set_documents(span, documents, scores=[score for _, score in fused])
            return documents
