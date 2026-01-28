from fastapi import APIRouter

from .documents import router as documents_router
from .indexing import router as indexing_router
from .policies import router as policies_router
from .search import router as search_router

# Re-exports for backward compatibility (tests patch these)
from agents.policies.agent.tools.retrieval.full_document_retrieval import (
    get_document_chunk_range,
    retrieve_full_document_async,
)

router = APIRouter()

router.include_router(documents_router)
router.include_router(indexing_router)
router.include_router(search_router)
router.include_router(policies_router)
