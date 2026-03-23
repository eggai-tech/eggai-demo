from unittest.mock import AsyncMock, patch

import pytest

from libraries.integrations.vector_store.config import VectorStoreBackend, VectorStoreConfig
from libraries.integrations.vector_store.schemas import PolicyDocument
from libraries.integrations.vector_store.store import InMemoryVectorStore
from libraries.integrations.vector_store.vespa_adapter import VespaVectorStore


def _create_policy_document(doc_id: str, chunk_index: int, embedding: list[float]) -> PolicyDocument:
    return PolicyDocument(
        id=f"{doc_id}_{chunk_index}",
        title="Policy Document",
        text=f"Chunk {chunk_index} text",
        category="auto",
        chunk_index=chunk_index,
        source_file="auto.md",
        page_numbers=[chunk_index + 1],
        page_range=str(chunk_index + 1),
        headings=["Coverage"],
        char_count=100,
        token_count=20,
        document_id=doc_id,
        previous_chunk_id=None,
        next_chunk_id=None,
        chunk_position=float(chunk_index),
        section_path=["Coverage"],
        embedding=embedding,
    )


@pytest.mark.asyncio
async def test_local_vector_store_refreshes_from_disk(tmp_path):
    config = VectorStoreConfig(
        backend=VectorStoreBackend.FAISS,
        faiss_path=tmp_path / "vector-store",
    )

    reader = InMemoryVectorStore(config=config)
    writer = InMemoryVectorStore(config=config)

    await writer.index_documents(
        [
            _create_policy_document("auto_policy", 0, [1.0, 0.0, 0.0]),
            _create_policy_document("auto_policy", 1, [0.8, 0.2, 0.0]),
        ]
    )

    field_results = await reader.search_documents(
        query='document_id:"auto_policy"',
        max_hits=10,
    )
    assert len(field_results) == 2

    vector_results = await reader.vector_search(
        query_embedding=[1.0, 0.0, 0.0],
        max_hits=1,
    )
    assert vector_results[0]["id"] == "auto_policy_0"


@pytest.mark.asyncio
async def test_vespa_adapter_routes_field_queries_through_custom_search():
    with patch("libraries.integrations.vespa.VespaClient") as mock_client_cls:
        mock_client = mock_client_cls.return_value
        mock_client.search_documents = AsyncMock(return_value=[])

        adapter = VespaVectorStore()

    with patch.object(
        adapter,
        "_search_documents_by_field",
        AsyncMock(return_value=[{"id": "doc1"}]),
    ) as mock_field_search:
        results = await adapter.search_documents(
            query='document_id:"auto_policy"',
            max_hits=5,
        )

    assert results == [{"id": "doc1"}]
    mock_field_search.assert_awaited_once()
    mock_client.search_documents.assert_not_awaited()
