from fastapi import APIRouter, Depends, HTTPException, Query

from agents.policies.agent.api.dependencies import get_search_service
from agents.policies.agent.api.models import (
    SearchRequest,
    SearchResponse,
)
from agents.policies.agent.api.validators import (
    validate_category,
    validate_query,
)
from agents.policies.agent.services.search_service import SearchService
from libraries.observability.logger import get_console_logger

logger = get_console_logger("policies_api_routes")

router = APIRouter()


@router.get("/kb/search")
async def search_documents(
    query: str = Query(..., description="Search query"),
    category: str | None = Query(None, description="Filter by category"),
    max_hits: int = Query(10, description="Maximum number of results", ge=1, le=100),
    search_service: SearchService = Depends(get_search_service),
):
    try:
        validated_query = validate_query(query)
        validated_category = validate_category(category) if category else None

        request = SearchRequest(
            query=validated_query,
            category=validated_category,
            max_hits=max_hits,
            search_type="keyword"
        )

        result = await search_service.search(request)

        return {
            "query": result.query,
            "category": result.category,
            "total_results": result.total_hits,
            "results": [
                {
                    "id": doc.id,
                    "title": doc.title,
                    "content": doc.text,
                    "category": doc.category,
                    "source_file": doc.source_file,
                    "document_id": doc.document_id,
                    "relevance": doc.relevance
                }
                for doc in result.documents
            ]
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Search error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Internal server error during search"
        )


@router.post("/kb/search/vector", response_model=SearchResponse)
async def vector_search(
    request: SearchRequest,
    search_service: SearchService = Depends(get_search_service),
):
    try:
        request.query = validate_query(request.query)
        if request.category:
            request.category = validate_category(request.category)

        result = await search_service.search(request)

        return SearchResponse(
            query=result.query,
            category=result.category,
            total_hits=result.total_hits,
            documents=result.documents,
            search_type=result.search_type
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Search error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Internal server error during search"
        )
