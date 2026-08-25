import logging

from db.database import Database
from monitoring import get_tracer, set_documents

from .embedder import MODEL_NAME, Embedder
from .schema import DOCUMENT_EMBEDDINGS_SCHEMA
from .utils import to_pgvector


logger = logging.getLogger(__name__)

tracer = get_tracer(__name__)


class VectorSearch:
    embedder: Embedder
    table: str
    index_name: str
    db: Database

    def __init__(
        self, embedder: Embedder, table: str, db: Database | None = None
    ) -> None:
        self.embedder = embedder
        self.table = table
        self.index_name = f"{table}_hnsw"
        self.db = db if db is not None else Database()

    def ensure_schema(self) -> None:
        """Create the vector table if it is not there yet."""
        with self.db as conn:
            with conn.cursor() as cur:
                cur.execute(DOCUMENT_EMBEDDINGS_SCHEMA.format(table=self.table))

    def index(self) -> None:
        """Create the pgvector HNSW index over the stored embeddings.

        This is the pgvector counterpart of fitting an in-memory index: rather
        than loading vectors into the process, it builds an approximate-nearest
        -neighbour structure inside postgres that the ``<=>`` operator can use.

        Optional. Without it, ``vector_search`` still works — postgres just
        scans every row, which is *exact* rather than approximate. At the
        current corpus size that scan measures ~15 ms against ~1.5 ms with the
        index, so the index trades exact recall for ~13 ms and an index roughly
        the size of the table. It earns its keep nearer 100k rows.
        """
        with self.db as conn:
            with conn.cursor() as cur:
                # An identifier cannot be bound as a parameter, so the name is
                # interpolated. It is configuration, not anything a searcher
                # supplies.
                cur.execute(
                    f"CREATE INDEX IF NOT EXISTS {self.index_name} "
                    f"ON {self.table} USING hnsw (embedding vector_cosine_ops)"
                )
        logger.info("HNSW index %s present on %s.", self.index_name, self.table)

    def vector_search(
        self, query: str, sql_query: str, num_results: int = 5
    ) -> list[dict]:
        """Semantic search over the stored embeddings, best matches first."""
        with tracer.start_as_current_span(
            "vector_search", openinference_span_kind="retriever"
        ) as span:
            span.set_input(query)
            span.set_attribute("retriever.num_results", num_results)
            query_vector = self.embedder.encode(query)
            results = self._vector_search(query_vector, num_results, sql_query)
            set_documents(span, results, scores=[row["score"] for row in results])
            return results

    def _vector_search(
        self, query_vector, num_results: int, sql_query: str
    ) -> list[dict]:
        """Return the nearest rows to an already-encoded query vector.

        The statement comes from the caller, so the shape of a result row is
        the caller's to decide — but it has to select a ``score`` column, which
        is what the span records. Three names are bound for it: ``query_vector``,
        ``num_results`` and ``model_name``. Unused ones are ignored, so a
        statement need only name the placeholders it wants.

        Scope the statement to ``model_name`` wherever the embeddings table
        records one: without it a corpus embedded under two models would mix
        incomparable vectors into a single ranking.
        """
        with self.db as conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql_query,
                    {
                        "query_vector": to_pgvector(query_vector),
                        "model_name": MODEL_NAME,
                        "num_results": num_results,
                    },
                )
                return cur.fetchall()
