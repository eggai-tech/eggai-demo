from typing import Any

import httpx

from libraries.observability.logger import get_console_logger

from .base import VectorStoreBase
from .query_utils import escape_yql_string, parse_field_query
from .schemas import PolicyDocument

logger = get_console_logger("vector_store.vespa")


class VespaVectorStore(VectorStoreBase):
    """Adapter wrapping the existing VespaClient to conform to VectorStoreBase."""

    def __init__(self):
        from libraries.integrations.vespa import VespaClient

        self._client = VespaClient()

    @property
    def vespa_app(self):
        return self._client.vespa_app

    async def check_connectivity(self) -> bool:
        return await self._client.check_connectivity()

    async def index_documents(self, documents: list[PolicyDocument]) -> dict[str, Any]:
        from libraries.integrations.vespa.schemas import (
            PolicyDocument as VespaPolicyDocument,
        )

        vespa_docs = [
            VespaPolicyDocument(**doc.model_dump()) for doc in documents
        ]
        return await self._client.index_documents(vespa_docs)

    async def search_documents(
        self,
        query: str,
        category: str | None = None,
        max_hits: int | None = None,
        ranking_profile: str | None = None,
    ) -> list[dict[str, Any]]:
        field_query = parse_field_query(query)
        if field_query:
            field, value = field_query
            return await self._search_documents_by_field(
                field=field,
                value=value,
                category=category,
                max_hits=max_hits,
                ranking_profile=ranking_profile,
            )

        return await self._client.search_documents(
            query=query, category=category, max_hits=max_hits, ranking_profile=ranking_profile
        )

    async def get_document_count(self) -> int:
        return await self._client.get_document_count()

    async def vector_search(
        self,
        query_embedding: list[float],
        category: str | None = None,
        max_hits: int | None = None,
        ranking_profile: str = "semantic",
    ) -> list[dict[str, Any]]:
        return await self._client.vector_search(
            query_embedding=query_embedding,
            category=category,
            max_hits=max_hits,
            ranking_profile=ranking_profile,
        )

    async def hybrid_search(
        self,
        query: str,
        query_embedding: list[float],
        category: str | None = None,
        max_hits: int | None = None,
        alpha: float = 0.7,
        ) -> list[dict[str, Any]]:
        return await self._client.hybrid_search(
            query=query,
            query_embedding=query_embedding,
            category=category,
            max_hits=max_hits,
            alpha=alpha,
        )

    async def _search_documents_by_field(
        self,
        field: str,
        value: str,
        category: str | None,
        max_hits: int | None,
        ranking_profile: str | None,
    ) -> list[dict[str, Any]]:
        max_hits = max_hits or self._client.config.max_hits
        ranking_profile = ranking_profile or self._client.config.ranking_profile

        conditions = [f'{field} contains "{escape_yql_string(value)}"']
        if category:
            conditions.append(f'category contains "{escape_yql_string(category)}"')

        yql = (
            f"select * from {self._client.config.schema_name} "
            f"where {' and '.join(conditions)}"
        )

        try:
            async with self._client.vespa_app.asyncio(
                connections=1,
                timeout=httpx.Timeout(self._client.config.vespa_timeout),
            ) as session:
                response = await session.query(
                    yql=yql,
                    hits=max_hits,
                    ranking=ranking_profile,
                )

                if not response.is_successful():
                    logger.error(
                        f"Field search failed: status {response.status_code}, response: {response.json}"
                    )
                    return []

                return self._client._extract_search_results(response)
        except Exception as e:
            logger.error(f"Field search error: {e}")
            return []

    async def get_document(self, doc_id: str, **kwargs) -> dict[str, Any] | None:
        try:
            async with self._client.vespa_app.asyncio(connections=1) as session:
                response = await session.get_data(
                    schema="policy_document", data_id=doc_id
                )
                if response.is_successful():
                    return response.json.get("fields", {})
                return None
        except Exception as e:
            logger.error(f"Get document error: {e}")
            return None

    async def delete_document(self, doc_id: str) -> bool:
        try:
            async with self._client.vespa_app.asyncio(connections=1) as session:
                response = await session.delete_data_point(
                    schema="policy_document", data_id=doc_id
                )
                return response.status_code == 200
        except Exception as e:
            logger.error(f"Delete document error: {e}")
            return False

    async def delete_all_documents(self) -> int:
        results = await self.search_documents(query="", max_hits=400)
        deleted = 0
        try:
            async with self._client.vespa_app.asyncio(connections=1) as session:
                for doc in results:
                    doc_id = doc.get("id")
                    if doc_id:
                        response = await session.delete_data_point(
                            schema="policy_document", data_id=doc_id
                        )
                        if response.status_code == 200:
                            deleted += 1
        except Exception as e:
            logger.error(f"Delete all error: {e}")
        return deleted
