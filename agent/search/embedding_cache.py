"""Text in, vectors out, without paying the model twice for the same text.

Embedding is the slow step of the pipeline, so every vector is kept under a
fingerprint of the text that produced it. Re-running against a database that
has barely changed is then almost free — which is what makes rebuilding the
whole index the affordable way to pick up edits and deletions.
"""

from __future__ import annotations

import hashlib

from db.database import Database

from .embedder import MODEL_NAME, Embedder
from .schema import EMBEDDING_CACHE_SCHEMA


class EmbeddingCache:
    def __init__(
        self, db: Database, embedder: Embedder, model: str = MODEL_NAME
    ) -> None:
        self.db = db
        self.embedder = embedder
        self.model = model

    def ensure_schema(self) -> None:
        """Create the cache table if it is not there yet."""
        with self.db as conn:
            with conn.cursor() as cur:
                cur.execute(EMBEDDING_CACHE_SCHEMA)

    def vectors_for(self, texts: list[str]) -> list[list[float]]:
        """One vector per text, in order, embedding only what is not cached."""
        hashes = [self._fingerprint(text) for text in texts]
        vectors = self._fetch(hashes)

        # Keyed by hash, so a text repeated across documents is embedded once.
        missing = {h: t for h, t in zip(hashes, texts) if h not in vectors}
        if missing:
            encoded = self.embedder.encode_batch(list(missing.values()))
            fresh = {h: [float(x) for x in v] for h, v in zip(missing, encoded)}
            self._store(fresh)
            vectors.update(fresh)

        return [vectors[h] for h in hashes]

    @staticmethod
    def _fingerprint(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    def _fetch(self, hashes: list[str]) -> dict[str, list[float]]:
        with self.db as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT content_hash, embedding FROM embedding_cache "
                    "WHERE embedding_model = %s AND content_hash = ANY(%s)",
                    (self.model, hashes),
                )
                return {row["content_hash"]: row["embedding"] for row in cur.fetchall()}

    def _store(self, vectors: dict[str, list[float]]) -> None:
        with self.db as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO embedding_cache "
                    "(content_hash, embedding_model, embedding) VALUES (%s, %s, %s) "
                    "ON CONFLICT DO NOTHING",
                    [(h, self.model, v) for h, v in vectors.items()],
                )
