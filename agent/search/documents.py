from .embedding_cache import EmbeddingCache
from .utils import to_pgvector


# Named by the caller, the way the vector table's own DDL is, so writer and
# reader can be pointed at the same place. Re-writing a chunk replaces the
# vector already stored for it, matching how the search index treats a re-write.
DOCUMENT_EMBEDDINGS_INSERT = """
INSERT INTO {table} (document_id, chunk_index, embedding)
VALUES (%s, %s, %s::vector)
ON CONFLICT (document_id, chunk_index) DO UPDATE
SET embedding = EXCLUDED.embedding
"""


class Documents:
    def __init__(self, db, query):
        self.db = db
        self.query = query
        self.documents = self._query_documents()

    def _query_documents(self) -> list[dict]:
        with self.db as conn:
            with conn.cursor() as cur:
                cur.execute(self.query)
                return cur.fetchall()

    def chunk_documents(
        self, field: str, size: int = 500, overlap: int = 50
    ) -> list[dict]:
        """Split `field` into pieces of `size`, each sharing `overlap` with the
        previous one, so a phrase straddling a boundary survives whole in at
        least one chunk. Keep `overlap` below `size`.

        Every other column is copied onto each chunk unchanged — build_index
        reads them straight off these dicts.
        """
        step = max(size - overlap, 1)
        chunks = []
        for doc in self.documents:
            text = doc.get(field) or ""
            # max(..., 1) so a document with no text still yields one chunk
            # instead of dropping out of the index entirely.
            for index, start in enumerate(range(0, max(len(text), 1), step)):
                chunks.append(
                    {**doc, field: text[start : start + size], "chunk_index": index}
                )
                # Stop once a chunk reaches the end, or overlap would keep
                # emitting tails already contained in this one.
                if start + size >= len(text):
                    break
        return chunks

    def embed_documents(
        self, chunks: list[dict], cache: EmbeddingCache, field: str, table: str
    ) -> None:
        """Store one vector per chunk in `table`, keyed by document and position.

        Takes the chunks rather than cutting its own, so the vectors written
        here describe exactly the rows the caller goes on to index. `field`
        must be the one they were cut on, or there is nothing there to embed,
        and `table` the one VectorSearch reads.
        """
        vectors = cache.vectors_for([chunk[field] for chunk in chunks])
        rows = [
            (chunk["id"], chunk["chunk_index"], to_pgvector(vector))
            for chunk, vector in zip(chunks, vectors)
        ]
        with self.db as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    DOCUMENT_EMBEDDINGS_INSERT.format(table=table), rows
                )


