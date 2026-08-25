from typing import Dict

from tqdm import tqdm

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

from monitoring import get_tracer, set_documents


tracer = get_tracer(__name__)


def _document_id(row: dict) -> str:
    """Stable id for a row, so re-writing it updates rather than duplicates.

    A chunk is identified by the document it came from and its position in it.
    """
    if "chunk_index" in row:
        return f"{row['id']}:{row['chunk_index']}"
    return str(row["id"])


class TextSearch:
    es_client: Elasticsearch
    index_name: str

    def __init__(
        self,
        es_client: Elasticsearch,
        index_name: str,
    ) -> None:
        self.es_client = es_client
        self.index_name = index_name

    def index(
        self,
        rows: list[dict],
        settings: Dict | None = None,
        mappings: Dict | None = None,
    ) -> None:
        """Index rows, creating the index if it is not there yet.

        Each row replaces whatever is stored under its id, so writing the same
        rows again updates them in place instead of duplicating them.

        `settings` and `mappings` describe the fields the rows carry, which is
        the caller's knowledge rather than this class's. They apply only when
        the index is created here: an index that already exists keeps the
        mappings it was born with, so changing them means dropping it first.
        Passing neither leaves Elasticsearch to infer the types it sees.
        """
        if not self.es_client.indices.exists(index=self.index_name):
            self.es_client.indices.create(
                index=self.index_name, settings=settings, mappings=mappings
            )

        actions = (
            {"_index": self.index_name, "_id": _document_id(row), "_source": row}
            for row in rows
        )
        bulk(self.es_client, tqdm(actions, total=len(rows)))
        self.es_client.indices.refresh(index=self.index_name)

    def text_search(self, search_query: Dict) -> list[dict]:
        """Full-text search over titles and lyrics, best matches first."""
        with tracer.start_as_current_span(
            "text_search", openinference_span_kind="retriever"
        ) as span:
            span.set_input(search_query)
            if (num_results := search_query.get("size")) is not None:
                span.set_attribute("retriever.num_results", num_results)
            response = self.es_client.search(
                index=self.index_name,
                body=search_query,
            )
            hits = response["hits"]["hits"]
            set_documents(
                span,
                [hit["_source"] for hit in hits],
                scores=[hit["_score"] for hit in hits],
            )
            return [hit["_source"] for hit in hits]
