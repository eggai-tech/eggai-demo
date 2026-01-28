from typing import Any

import httpx
from vespa.io import VespaQueryResponse

from libraries.observability.logger import get_console_logger
from libraries.observability.tracing import create_tracer

logger = get_console_logger("vespa_client")
tracer = create_tracer("vespa", "client")


class VespaSearchMixin:
    def _extract_search_results(
        self, response: VespaQueryResponse
    ) -> list[dict[str, Any]]:
        results = []
        root_data = response.json.get("root", {})
        children = root_data.get("children", [])

        for child in children:
            fields = child.get("fields", {})
            results.append(
                {
                    "id": fields.get("id", ""),
                    "title": fields.get("title", ""),
                    "text": fields.get("text", ""),
                    "category": fields.get("category", ""),
                    "chunk_index": fields.get("chunk_index", 0),
                    "source_file": fields.get("source_file", ""),
                    "relevance": child.get("relevance", 0.0),
                    "page_numbers": fields.get("page_numbers", []),
                    "page_range": fields.get("page_range"),
                    "headings": fields.get("headings", []),
                    "char_count": fields.get("char_count", 0),
                    "token_count": fields.get("token_count", 0),
                    "document_id": fields.get("document_id"),
                    "previous_chunk_id": fields.get("previous_chunk_id"),
                    "next_chunk_id": fields.get("next_chunk_id"),
                    "chunk_position": fields.get("chunk_position"),
                    "section_path": fields.get("section_path", []),
                }
            )

        return results

    @tracer.start_as_current_span("search_documents")
    async def search_documents(
        self,
        query: str,
        category: str | None = None,
        max_hits: int | None = None,
        ranking_profile: str | None = None,
    ) -> list[dict[str, Any]]:
        max_hits = max_hits or self.config.max_hits
        ranking_profile = ranking_profile or self.config.ranking_profile

        logger.info(
            f"Searching for: '{query}', category: '{category}', max_hits: {max_hits}"
        )

        if query:
            yql_conditions = ["userInput(@query)"]
        else:
            yql_conditions = ["true"]

        if category:
            yql_conditions.append(f'category contains "{category}"')

        yql = f"select * from {self.config.schema_name} where {' and '.join(yql_conditions)}"

        try:
            async with self.vespa_app.asyncio(
                connections=1, timeout=httpx.Timeout(self.config.vespa_timeout)
            ) as session:
                query_params = {
                    "yql": yql,
                    "hits": max_hits,
                    "ranking": ranking_profile,
                }

                if query:
                    query_params["query"] = query

                response: VespaQueryResponse = await session.query(**query_params)

                if not response.is_successful():
                    logger.error(
                        f"Search failed: status {response.status_code}, "
                        f"response: {response.json}"
                    )
                    return []

                results = self._extract_search_results(response)
                logger.info(f"Found {len(results)} documents")
                return results

        except Exception as e:
            logger.error(f"Search error: {e}")
            return []

    @tracer.start_as_current_span("get_document_count")
    async def get_document_count(self) -> int:
        try:
            async with self.vespa_app.asyncio(connections=1) as session:
                response = await session.query(
                    yql=f"select * from {self.config.schema_name} where true limit 0",
                    hits=0,
                )

                if response.is_successful():
                    total_count = (
                        response.json.get("root", {})
                        .get("fields", {})
                        .get("totalCount", 0)
                    )
                    logger.info(f"Total documents in index: {total_count}")
                    return total_count
                else:
                    logger.error(
                        f"Failed to get document count: {response.status_code}"
                    )
                    return 0

        except Exception as e:
            logger.error(f"Error getting document count: {e}")
            return 0

    @tracer.start_as_current_span("vector_search")
    async def vector_search(
        self,
        query_embedding: list[float],
        category: str | None = None,
        max_hits: int | None = None,
        ranking_profile: str = "semantic",
    ) -> list[dict[str, Any]]:
        max_hits = max_hits or self.config.max_hits

        logger.info(
            f"Performing vector search, category: '{category}', max_hits: {max_hits}"
        )

        yql_conditions = [
            f"{{targetHits: {max_hits}}}nearestNeighbor(embedding, query_embedding)"
        ]

        if category:
            yql_conditions.append(f'category contains "{category}"')

        yql = f"select * from {self.config.schema_name} where {' and '.join(yql_conditions)}"

        try:
            async with self.vespa_app.asyncio(
                connections=1, timeout=httpx.Timeout(self.config.vespa_timeout)
            ) as session:
                embedding_list = [float(val) for val in query_embedding]

                response = await session.query(
                    yql=yql,
                    ranking=ranking_profile,
                    hits=max_hits,
                    body={
                        "input.query(query_embedding)": embedding_list,
                        "presentation.timing": True,
                    },
                )

                if not response.is_successful():
                    logger.error(
                        f"Vector search failed: status {response.status_code}, "
                        f"response: {response.json}"
                    )
                    return []

                results = self._extract_search_results(response)
                logger.info(f"Vector search found {len(results)} documents")
                return results

        except Exception as e:
            logger.error(f"Vector search error: {e}")
            return []

    @tracer.start_as_current_span("hybrid_search")
    async def hybrid_search(
        self,
        query: str,
        query_embedding: list[float],
        category: str | None = None,
        max_hits: int | None = None,
        alpha: float = 0.7,
    ) -> list[dict[str, Any]]:
        max_hits = max_hits or self.config.max_hits

        logger.info(
            f"Performing hybrid search: '{query}', alpha: {alpha}, category: '{category}'"
        )

        base_condition = (
            f"{{targetHits: {max_hits * 2}}}nearestNeighbor(embedding, query_embedding)"
        )

        if category:
            yql = f'select * from {self.config.schema_name} where {base_condition} and category contains "{category}"'
        else:
            yql = f"select * from {self.config.schema_name} where {base_condition}"

        try:
            async with self.vespa_app.asyncio(
                connections=1, timeout=httpx.Timeout(self.config.vespa_timeout)
            ) as session:
                embedding_list = [float(val) for val in query_embedding]

                body = {
                    "input.query(query_embedding)": embedding_list,
                    "input.query(alpha)": alpha,
                    "presentation.timing": True,
                }

                query_params = {
                    "yql": yql,
                    "ranking": "hybrid",
                    "hits": max_hits,
                    "body": body,
                }

                if query:
                    query_params["query"] = query

                response = await session.query(**query_params)

                if not response.is_successful():
                    logger.error(
                        f"Hybrid search failed: status {response.status_code}, "
                        f"response: {response.json}"
                    )
                    return []

                results = self._extract_search_results(response)
                logger.info(f"Hybrid search found {len(results)} documents")
                return results

        except Exception as e:
            logger.error(f"Hybrid search error: {e}")
            return []
