"""The tables the search package owns, one statement per store.

Kept as strings rather than a .sql file so the vector table can be named by
whoever creates it: ``VectorSearch`` is configured with a table, and its DDL
has to follow that configuration rather than hard-coding a name beside it.
"""

# Cache of text -> embedding, so rebuilding the index only pays the model for
# chunks whose text actually changed.
#
# Keyed by the model as well as the text: a different model produces different
# numbers for the same words, so one model's rows must never be served to
# another's request.
#
# float8[] rather than pgvector: nothing here searches by similarity, this is a
# plain key-value lookup, and psycopg maps a Postgres array straight to a list
# without registering an adapter.
EMBEDDING_CACHE_SCHEMA = """
CREATE TABLE IF NOT EXISTS embedding_cache (
    content_hash    TEXT               NOT NULL,
    embedding_model TEXT               NOT NULL,
    embedding       DOUBLE PRECISION[] NOT NULL,
    PRIMARY KEY (content_hash, embedding_model)
);
"""

# The vector store: one row per chunk, ready to be searched by meaning.
#
# Keyed by (document_id, chunk_index) -- the same pair the search index builds
# its _id from, so a row here and a document there name the same chunk.
#
# pgvector here, unlike the cache above, because this table is queried with the
# <=> distance operator. That is the whole reason the extension exists.
#
# Formatted with the table name. The HNSW index over it is not created here;
# that is VectorSearch.index(), which is optional and named off the same table.
DOCUMENT_EMBEDDINGS_SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS {table} (
    document_id BIGINT      NOT NULL,
    chunk_index INTEGER     NOT NULL,
    embedding   vector(768) NOT NULL,
    PRIMARY KEY (document_id, chunk_index)
);
"""
