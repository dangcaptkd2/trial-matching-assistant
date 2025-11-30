"""Elasticsearch utilities for indexing clinical trials (BM25 baseline)."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

from elasticsearch import Elasticsearch, helpers


from src.config.settings import settings


class ElasticsearchService:
    """Thin wrapper around Elasticsearch Python client."""

    def __init__(self, url: Optional[str] = None):
        self.url = url or settings.elasticsearch_url or "http://localhost:9200"
        # Elastic 7.x default
        self.client = Elasticsearch(self.url)

    def create_index(
        self,
        index_name: str,
        settings_body: Dict[str, Any],
        mappings_body: Dict[str, Any],
    ) -> None:
        if self.client.indices.exists(index=index_name):
            return

        self.client.indices.create(
            index=index_name,
            body={
                "settings": settings_body,
                "mappings": mappings_body,
            },
        )

    def document_exists(self, index_name: str, doc_id: str) -> bool:
        try:
            return self.client.exists(index=index_name, id=doc_id)
        except Exception:
            return False

    def index_document(
        self, index_name: str, doc_id: str, body: Dict[str, Any], op_type: str = "index"
    ) -> None:
        self.client.index(index=index_name, id=doc_id, body=body, op_type=op_type)

    def bulk_index(self, actions: Iterable[Dict[str, Any]]) -> None:
        helpers.bulk(self.client, actions)  # type: ignore
