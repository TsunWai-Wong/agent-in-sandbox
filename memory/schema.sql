-- The memory store: an append-only log of operations, never a table of rows
-- that get edited.
--
-- Nothing in here is ever UPDATEd or DELETEd. Changing a memory inserts a
-- successor pointing back at what it replaced; forgetting one inserts a
-- tombstone doing the same. That buys two things an in-place table cannot:
-- a bad write is recoverable, and "the user moved from Berlin to Munich" stays
-- readable as a history rather than collapsing into a single current value.
--
-- The consequence is that `status` and `superseded_by` are not columns. They
-- are derived — a row is active when nothing supersedes it and it has not
-- expired — which is the same reasoning the run ledger applies to its own
-- counters: if it can be read off the log, storing it beside the log just
-- gives you two answers that drift.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE SCHEMA IF NOT EXISTS memory;

CREATE TABLE IF NOT EXISTS memory.user_memories (
    id              BIGSERIAL PRIMARY KEY,
    user_id         TEXT        NOT NULL,
    op              TEXT        NOT NULL,
    memory_type     TEXT        NOT NULL,
    content         TEXT,
    confidence      SMALLINT    NOT NULL,
    supersedes_id   BIGINT      REFERENCES memory.user_memories(id),
    source_run_id   TEXT        NOT NULL DEFAULT '',
    -- 768 dimensions: EmbeddingGemma-300M's output width. Nullable, so a store
    -- running without the model still records memories and simply falls back
    -- to keyword retrieval.
    embedding       vector(768),
    -- Which model produced that vector. One column, and it is what stops a
    -- corpus embedded under two models from fusing incomparable distances into
    -- one ranking after a model switch.
    embedding_model TEXT        NOT NULL DEFAULT '',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at      TIMESTAMPTZ,

    CONSTRAINT op_is_known
        CHECK (op IN ('add', 'update', 'invalidate')),
    CONSTRAINT memory_type_is_known
        CHECK (memory_type IN ('directive', 'profile', 'current', 'event')),
    CONSTRAINT confidence_in_range
        CHECK (confidence BETWEEN 1 AND 3),
    -- A tombstone carries no content; everything else must carry some.
    CONSTRAINT content_matches_op
        CHECK ((op = 'invalidate') = (content IS NULL)),
    -- An add creates; an update or an invalidate must say what it replaces.
    CONSTRAINT supersedes_matches_op
        CHECK ((op = 'add') = (supersedes_id IS NULL)),
    -- History is a chain, never a fork. Without this, two concurrent writers
    -- can both supersede the same row and the active set silently loses one of
    -- them. NULLs stay distinct in Postgres, so plain adds are unaffected.
    CONSTRAINT one_successor_per_memory
        UNIQUE (supersedes_id)
);

CREATE INDEX IF NOT EXISTS user_memories_user_type
    ON memory.user_memories (user_id, memory_type);

-- What "active" means, in one place.
--
-- Excluded: tombstones themselves, anything a later row supersedes, and
-- anything past its expiry. A tombstone therefore removes two rows from this
-- view at once — itself by op, and its target by the NOT EXISTS.
CREATE OR REPLACE VIEW memory.active_memories AS
SELECT m.*
FROM memory.user_memories m
WHERE m.op <> 'invalidate'
  AND NOT EXISTS (
      SELECT 1
      FROM memory.user_memories s
      WHERE s.supersedes_id = m.id
  )
  AND (m.expires_at IS NULL OR m.expires_at > now());

-- Optional, and genuinely optional at this size. A user's active memories
-- number in the dozens to low hundreds, where an exact scan costs less than a
-- millisecond and is exact rather than approximate. Build it when a single
-- user's store runs to tens of thousands of rows, not before.
--
--   CREATE INDEX user_memories_hnsw ON memory.user_memories
--       USING hnsw (embedding vector_cosine_ops);
