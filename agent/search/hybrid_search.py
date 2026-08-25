from monitoring import get_tracer, set_documents

from .text_search import TextSearch
from .vector_search import VectorSearch

tracer = get_tracer(__name__)

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

    def hybrid_search(
        self,
        query: str,
        text_query: dict,
        vector_query: str,
        num_results: int = 10,
    ) -> list[dict]:
        """Search the corpus with combined full-text and semantic search.

        Runs Elasticsearch full-text search and vector similarity search,
        then merges both rankings with Reciprocal Rank Fusion.

        Args:
            query: Natural-language description of the documents to find.
            text_query: Elasticsearch query body. Its `size` is replaced with
                the pool size, so both rankings are fused over equal depth.
            vector_query: Nearest-neighbour statement for the vector store.
            num_results: Maximum number of documents to return.
        """
        with tracer.start_as_current_span(
            "hybrid_search", openinference_span_kind="chain"
        ) as span:
            span.set_input(query)
            pool_size = max(2 * num_results, 10)
            span.set_attribute("retriever.pool_size", pool_size)

            text_results = self.text_searcher.text_search(
                {**text_query, "size": pool_size}
            )
            vector_results = self.vector_searcher.vector_search(
                query, vector_query, num_results=pool_size
            )

            fused = self._rrf(
                [text_results, vector_results], num_results=num_results
            )
            documents = [document for document, _ in fused]
            set_documents(span, documents, scores=[score for _, score in fused])
            return documents