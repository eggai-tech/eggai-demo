from fastapi import APIRouter, Depends, HTTPException, Query

from agents.policies.agent.api.dependencies import get_document_service
from agents.policies.agent.api.models import (
    CategoryStats,
    FullDocumentResponse,
    PolicyDocument,
)
from agents.policies.agent.api.validators import (
    validate_category,
    validate_document_id,
)
from agents.policies.agent.services.document_service import DocumentService
from libraries.observability.logger import get_console_logger

logger = get_console_logger("policies_api_routes")

router = APIRouter()


@router.get("/health")
async def health_check():
    return {"status": "healthy", "service": "policies-agent", "version": "1.0.0"}


@router.get("/kb/documents", response_model=list[PolicyDocument])
async def list_documents(
    category: str | None = Query(None, description="Filter by category"),
    limit: int = Query(20, description="Number of documents to return", ge=1, le=100),
    offset: int = Query(0, description="Offset for pagination", ge=0),
    document_service: DocumentService = Depends(get_document_service),
):
    try:
        validated_category = validate_category(category)

        documents = await document_service.list_documents(
            category=validated_category,
            limit=limit,
            offset=offset
        )
        return documents
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"List documents error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Internal server error while listing documents"
        )


@router.get("/kb/categories", response_model=list[CategoryStats])
async def get_categories(
    document_service: DocumentService = Depends(get_document_service),
):
    try:
        stats = await document_service.get_categories_stats()
        return [CategoryStats(**stat) for stat in stats]
    except Exception as e:
        logger.error(f"Get categories error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Internal server error while retrieving categories"
        )


@router.get("/kb/documents/{doc_id}", response_model=PolicyDocument)
async def get_document(
    doc_id: str,
    document_service: DocumentService = Depends(get_document_service),
):
    try:
        validated_doc_id = validate_document_id(doc_id)

        document = await document_service.get_document_by_id(validated_doc_id)

        if not document:
            raise HTTPException(
                status_code=404, detail=f"Document not found: {doc_id}"
            )

        return document
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get document error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Internal server error while retrieving document"
        )


@router.get("/kb/documents/{document_id}/full", response_model=FullDocumentResponse)
async def get_full_document(document_id: str):
    from agents.policies.agent.api import routes

    try:
        validated_doc_id = validate_document_id(document_id)

        full_doc = await routes.retrieve_full_document_async(validated_doc_id)

        if not full_doc:
            raise HTTPException(
                status_code=404, detail=f"Document not found: {document_id}"
            )

        return full_doc
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get full document error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Internal server error while retrieving full document"
        )


@router.get("/kb/documents/{document_id}/chunks", response_model=list[PolicyDocument])
async def get_document_chunks(
    document_id: str,
    document_service: DocumentService = Depends(get_document_service),
):
    try:
        validated_doc_id = validate_document_id(document_id)

        all_documents = await document_service.list_documents(limit=1000)
        chunks = [doc for doc in all_documents if doc.document_id == validated_doc_id]

        if not chunks:
            raise HTTPException(status_code=404, detail="Document not found")

        return chunks
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get document chunks error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Internal server error while retrieving document chunks"
        )


@router.get("/kb/documents/{document_id}/range", response_model=FullDocumentResponse)
async def get_document_range(
    document_id: str,
    start_chunk: int = Query(0, description="Starting chunk index", ge=0),
    end_chunk: int | None = Query(None, description="Ending chunk index (inclusive)"),
):
    from agents.policies.agent.api import routes

    try:
        validated_doc_id = validate_document_id(document_id)

        if end_chunk is not None and end_chunk < start_chunk:
            raise HTTPException(
                status_code=400,
                detail="End chunk must be greater than or equal to start chunk"
            )

        doc_range = routes.get_document_chunk_range(
            validated_doc_id, start_chunk, end_chunk
        )

        if not doc_range:
            raise HTTPException(
                status_code=404,
                detail=f"Document not found or invalid range: {document_id}",
            )

        return doc_range
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get document range error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Internal server error while retrieving document range"
        )
