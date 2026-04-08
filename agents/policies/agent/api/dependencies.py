from functools import lru_cache
from typing import Annotated

from fastapi import Depends
from sentence_transformers import SentenceTransformer

from agents.policies.agent.config import settings
from agents.policies.agent.services.document_service import DocumentService
from agents.policies.agent.services.reindex_service import ReindexService
from agents.policies.agent.services.search_service import SearchService
from libraries.integrations.vector_store import VectorStoreBase, create_vector_store
from libraries.observability.logger import get_console_logger

logger = get_console_logger("policies_api_dependencies")


@lru_cache
def get_vector_store() -> VectorStoreBase:
    logger.info("Creating vector store instance")
    return create_vector_store()


@lru_cache
def get_embedding_model() -> SentenceTransformer:
    logger.info(f"Loading embedding model: {settings.embedding_model}")
    return SentenceTransformer(settings.embedding_model)


def get_document_service(
    vector_store: Annotated[VectorStoreBase, Depends(get_vector_store)]
) -> DocumentService:
    return DocumentService(vector_store)


def get_search_service(
    vector_store: Annotated[VectorStoreBase, Depends(get_vector_store)],
    embedding_model: Annotated[SentenceTransformer, Depends(get_embedding_model)]
) -> SearchService:
    return SearchService(vector_store, embedding_model)


def get_reindex_service(
    vector_store: Annotated[VectorStoreBase, Depends(get_vector_store)]
) -> ReindexService:
    return ReindexService(vector_store)
