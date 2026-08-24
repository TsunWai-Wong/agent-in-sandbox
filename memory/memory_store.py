"""UserMemoryStore: the only thing that writes to the memory log.

Three properties this class owns, so no caller has to remember them:

Every statement is scoped to a user. `user_id` is in the WHERE clause of every
read and the ownership check of every write, so a memory id alone is never
enough to touch someone else's row.

Nothing is ever updated or deleted. `update()` inserts a successor and
`delete()` inserts a tombstone; the row they act on stays exactly as written.
`status` and `superseded_by` are read out of the log by the active_memories view
rather than stored beside it.

Expiry is derived, never supplied — the one caller that would pass it is a
language model, and the number it produces means nothing.

What this class does NOT do is decide between adding and updating. That is
memory_writer.py's job, which is what keeps this class free of a model entirely.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

from db.database import Database
from monitoring import get_tracer

from .embedder import DOCUMENT_PROMPT, MODEL_NAME, QUERY_PROMPT, Embedder, to_pgvector
from .types import (
    INJECTED_TYPES,
    MIN_INJECT_CONFIDENCE,
    MemoryContent,
    MemoryType,
    Operation,
    expires_at,
)


logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)

SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"

# Only these may be carried into a successor row. Everything else — id, user_id,
# op, created_at, supersedes_id — is the log's to decide, and a patch that could
# set them would let a caller rewrite provenance.
PATCHABLE = frozenset({"content", "memory_type", "confidence"})

DEFAULT_SEARCH_LIMIT = 8

# What the per-request recall block asks for. Smaller than a deliberate search:
# it rides along on every model call, so it is paying rent in the prompt.
DEFAULT_RECALL_LIMIT = 5
DEFAULT_NEIGHBOURS = 5

# Cosine similarity below which a "neighbour" is not about the same thing. Only
# decides whether reconciliation is worth a model call, so a generous threshold
# costs a little money and a stingy one costs a duplicate row. Erring generous.
NEIGHBOUR_THRESHOLD = 0.55

_COLUMNS = """
    id, user_id, op, memory_type, content, confidence,
    supersedes_id, source_run_id, embedding_model, created_at, expires_at
"""

INSERT_SQL = f"""
INSERT INTO memory.user_memories (
    user_id, op, memory_type, content, confidence,
    supersedes_id, source_run_id, embedding, embedding_model, expires_at
) VALUES (
    %(user_id)s, %(op)s, %(memory_type)s, %(content)s, %(confidence)s,
    %(supersedes_id)s, %(source_run_id)s, %(embedding)s::vector,
    %(embedding_model)s, %(expires_at)s
)
RETURNING {_COLUMNS}
"""

ACTIVE_SQL = f"""
SELECT {_COLUMNS}
FROM memory.active_memories
WHERE user_id = %(user_id)s
  AND (%(types)s::text[] IS NULL OR memory_type = ANY(%(types)s))
  AND confidence >= %(min_confidence)s
ORDER BY memory_type, created_at DESC
"""

GET_SQL = f"""
SELECT {_COLUMNS}
FROM memory.active_memories
WHERE user_id = %(user_id)s AND id = %(id)s
"""

# Nearest first. The distance scan covers one user's rows — dozens to low
# hundreds — so it is exact and fast enough that the HNSW index in schema.sql
# stays commented out.
VECTOR_SQL = f"""
SELECT {_COLUMNS}, 1 - (embedding <=> %(query_vector)s::vector) AS score
FROM memory.active_memories
WHERE user_id = %(user_id)s
  AND embedding IS NOT NULL
  AND embedding_model = %(embedding_model)s
ORDER BY embedding <=> %(query_vector)s::vector
LIMIT %(limit)s
"""


class UserMemoryStore:
    db: Database
    embedder: Embedder | None

    def __init__(
        self, db: Database | None = None, embedder: Embedder | None = None
    ) -> None:
        self.db = db if db is not None else Database()
        # Optional throughout. Without it the store still records and recalls;
        # it just matches on words instead of meaning.
        self.embedder = embedder

    # -- setup ---------------------------------------------------------------

    def ensure_schema(self) -> None:
        """Create the schema, table and view if they are not there yet.

        Every statement in schema.sql is IF NOT EXISTS or CREATE OR REPLACE, so
        running this against an existing database is a no-op.
        """
        with self.db as conn:
            with conn.cursor() as cur:
                cur.execute(SCHEMA_PATH.read_text())
        logger.info("Memory schema present.")

    # -- writes --------------------------------------------------------------

    def add(self, user_id: str, data: dict) -> dict:
        """Record a new memory. Validates `data` as a MemoryContent first."""
        draft = MemoryContent.model_validate(data)
        return self._insert(
            user_id,
            Operation.ADD,
            draft,
            source_run_id=data.get("source_run_id", ""),
        )

    def update(self, user_id: str, memory_id: int, data: dict) -> dict:
        """Supersede a memory with a revised version of it.

        The original row is untouched, which is what makes a wrong update
        recoverable and "Berlin, then Munich" readable as a history.

        Raises KeyError if the target is not an active memory of this user, so a
        stale or borrowed id fails rather than writing an orphan.
        """
        target = self.get(user_id, memory_id)
        if target is None:
            raise KeyError(f"no active memory {memory_id} for this user")

        unknown = set(data) - PATCHABLE
        if unknown:
            raise ValueError(f"fields are not patchable: {sorted(unknown)}")

        merged = {field: target[field] for field in PATCHABLE}
        merged.update(data)
        return self._insert(
            user_id,
            Operation.UPDATE,
            MemoryContent.model_validate(merged),
            supersedes_id=memory_id,
            source_run_id=target.get("source_run_id", ""),
        )

    def delete(self, user_id: str, memory_id: int, source_run_id: str = "") -> dict:
        """Retire a memory that is no longer true and has no replacement.

        A tombstone, not a DELETE: the row it points at drops out of the active
        view while both stay on the log. "I don't work with Alice anymore" is
        frequently worth keeping as history.
        """
        target = self.get(user_id, memory_id)
        if target is None:
            raise KeyError(f"no active memory {memory_id} for this user")

        # A tombstone has no content of its own, but the CHECK constraints still
        # want a type and a confidence, and the target's are the honest ones.
        return self._row(
            user_id=user_id,
            op=Operation.INVALIDATE.value,
            memory_type=target["memory_type"],
            content=None,
            confidence=target["confidence"],
            supersedes_id=memory_id,
            source_run_id=source_run_id,
        )

    def _insert(
        self,
        user_id: str,
        op: Operation,
        draft: MemoryContent,
        supersedes_id: int | None = None,
        source_run_id: str = "",
    ) -> dict:
        vector = self._embed(draft.content, DOCUMENT_PROMPT)
        return self._row(
            user_id=user_id,
            op=op.value,
            memory_type=draft.memory_type.value,
            content=draft.content,
            confidence=draft.confidence,
            supersedes_id=supersedes_id,
            source_run_id=source_run_id,
            embedding=vector,
            embedding_model=MODEL_NAME if vector else "",
            # Derived here and nowhere else. Never a caller's to pass.
            expires_at=expires_at(draft.memory_type, draft.confidence),
        )

    def _row(self, **params) -> dict:
        """The only INSERT. Every write in this class is one of these.

        Defaults cover the columns a tombstone leaves empty, so the caller
        names only what it actually decides.
        """
        return self._one(
            INSERT_SQL,
            {
                "supersedes_id": None,
                "source_run_id": "",
                "embedding": None,
                "embedding_model": "",
                "expires_at": None,
                **params,
            },
        )

    # -- reads ---------------------------------------------------------------

    def get(self, user_id: str, memory_id: int) -> dict | None:
        """One active memory by id, or None. Scoped to the user, always."""
        return self._one(GET_SQL, {"user_id": user_id, "id": memory_id})

    def active(
        self,
        user_id: str,
        types: Iterable[MemoryType] | None = None,
        min_confidence: int = 1,
    ) -> list[dict]:
        """Every memory currently in force, optionally filtered by type.

        What the always-injected directive block is built from, so it returns
        whole rows in a stable order rather than a ranking.
        """
        return self._all(
            ACTIVE_SQL,
            {
                "user_id": user_id,
                "types": [MemoryType(t).value for t in types] if types else None,
                "min_confidence": min_confidence,
            },
        )

    def search(
        self,
        user_id: str,
        query: str,
        limit: int = DEFAULT_SEARCH_LIMIT,
        threshold: float = 0.0,
        with_directives: bool = False,
    ) -> list[dict]:
        """What is known about this user, relevant to `query`. Rows, not text.

        The vector half returns nothing when there is no embedder to ask — the
        cost of having a single retrieval path.

        `with_directives` adds the standing preferences on top, and it is not a
        convenience. A directive has no reason to *resemble* the question it
        applies to: "answer concisely" will never come back as a neighbour of
        "where do I live?". Recall passes it so the agent reads them on every
        turn; they come first, and they survive a missing embedder because
        active() is plain SQL.

        `threshold` is what separates the two callers. Recall wants whatever is
        closest and takes it. Reconciliation wants only rows near enough to be
        *the same fact*, since those are the only ones a verdict may act on, so
        it passes NEIGHBOUR_THRESHOLD and accepts an empty list.
        """
        with tracer.start_as_current_span(
            "memory_search", openinference_span_kind="retriever"
        ) as span:
            span.set_input(query)
            span.set_attribute("retriever.num_results", limit)

            rows = self._vector_search(user_id, query, limit) if query else []
            if threshold:
                rows = [r for r in rows if r.get("score", 0) >= threshold]

            if with_directives:
                standing = self.active(
                    user_id,
                    types=INJECTED_TYPES,
                    min_confidence=MIN_INJECT_CONFIDENCE,
                )
                # Deduped by id, not by content: the same row reached by both
                # routes must appear once, and id is what identifies it.
                seen = {row["id"] for row in standing}
                rows = standing + [r for r in rows if r["id"] not in seen]

            # Worth a span attribute now that a missing embedder means zero
            # results rather than worse ones: it is the difference between
            # "nothing is stored" and "nothing can be read".
            span.set_attribute("retriever.embedded", self.embedder is not None)
            span.set_attribute("retriever.hits", len(rows))
            return rows

    # -- retrieval mechanics -------------------------------------------------

    def _vector_search(self, user_id: str, query: str, limit: int) -> list[dict]:
        """Nearest neighbours, or nothing when there is no embedder to ask."""
        vector = self._embed(query, QUERY_PROMPT)
        if vector is None:
            return []
        return self._all(
            VECTOR_SQL,
            {
                "user_id": user_id,
                "query_vector": vector,
                # Scoped to the model that produced the stored vectors, so a
                # half-migrated store ranks on comparable distances instead of
                # mixing two embedding spaces.
                "embedding_model": MODEL_NAME,
                "limit": limit,
            },
        )

    # -- the two primitives ---------------------------------------------------
    # Every read and write above is one of these. Materialised inside the block,
    # because db closes the connection on exit and a lazy cursor handed outside
    # it reads from a closed one.

    def _one(self, sql: str, params: dict) -> dict | None:
        with self.db as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                return cur.fetchone()

    def _all(self, sql: str, params: dict) -> list[dict]:
        with self.db as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                return cur.fetchall()

    def _embed(self, text: str, prompt: str) -> str | None:
        """Embed one string, or return None if that is not possible right now.

        Never raises. A model that fails to load, or an ONNX session that dies
        mid-run, must not take a memory write down with it — the row is worth
        having without its vector, and the vector can be backfilled.

        `prompt` is the caller's to choose: queries and documents must not share
        one. See embedder.py.
        """
        if self.embedder is None:
            return None
        try:
            return to_pgvector(self.embedder.encode(text, prompt))
        except Exception:
            logger.exception("Embedding failed; storing without a vector.")
            return None
