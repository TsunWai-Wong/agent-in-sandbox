"""Wiring for the retrieval pipeline: source rows in, searchable stores out.

Holds the configuration the stages disagree about if left to themselves — one
field to chunk and embed on, one table for the vectors, one name for the text
index — so a single place decides them and the stages cannot drift apart.
"""

import logging

from elasticsearch import Elasticsearch

from db.database import Database

from .documents import Documents
from .embedder import Embedder
from .embedding_cache import EmbeddingCache
from .hybrid_search import HybridSearch
from .text_search import TextSearch
from .vector_search import VectorSearch


logger = logging.getLogger(__name__)


class RAGPipeline:
    db: Database
    document_sql: str
    es_client: Elasticsearch
    text_index_name: str
    text_index_mapping: dict
    embedder: Embedder
    vector_table_name: str
    field_to_embed: str
    to_build_vector_index: bool
    text_searcher: TextSearch
    vector_searcher: VectorSearch
    hybrid_searcher: HybridSearch

    def __init__(
        self,
        db: Database,
        document_sql: str,
        es_client: Elasticsearch,
        text_index_name: str,
        text_index_mapping: dict,
        embedder: Embedder,
        vector_table_name: str,
        field_to_embed: str,
        to_build_vector_index: bool = False
    ) -> None:
        self.db = db
        self.document_sql = document_sql
        self.es_client = es_client
        self.text_index_name = text_index_name
        self.text_index_mapping = text_index_mapping
        self.embedder = embedder
        self.vector_table_name = vector_table_name
        self.field_to_embed = field_to_embed
        self.to_build_vector_index = to_build_vector_index

        # Built here, not in run(): both are cheap references, and searching
        # must not require having just indexed. Holding them is also what keeps
        # the caller from naming the index or the table a second time.
        self.text_searcher = TextSearch(es_client, text_index_name)
        self.vector_searcher = VectorSearch(embedder, vector_table_name, db)
        self.hybrid_searcher = HybridSearch(self.text_searcher, self.vector_searcher)

    def run(self) -> None:
        """Read the documents, cut them up, and fill both stores from the same
        chunks.

        One chunking pass feeds the embeddings and the text index alike, so a
        row in one names the same piece of text as the row beside it in the
        other. Both schemas are created if absent, so a fresh database needs no
        setup step first.

        Additive rather than a rebuild: a document deleted at the source keeps
        its rows in both stores, and the mapping applies only if the text index
        does not exist yet.
        """
        documents = Documents(self.db, self.document_sql)
        chunks = documents.chunk_documents(self.field_to_embed)
        logger.info("Embedding %d chunks into %s.", len(chunks), self.vector_table_name)

        cache = EmbeddingCache(self.db, self.embedder)
        cache.ensure_schema()

        self.vector_searcher.ensure_schema()
        documents.embed_documents(
            chunks, cache, self.field_to_embed, self.vector_table_name
        )

        if self.to_build_vector_index:
            logger.info("Building the vector index. This is the slow step.")
            self.vector_searcher.index()

        logger.info("Indexing %d chunks into %s.", len(chunks), self.text_index_name)
        self.text_searcher.index(chunks, mappings=self.text_index_mapping)
