from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile

from agents.policies.agent.api.dependencies import (
    get_document_service,
    get_reindex_service,
)
from agents.policies.agent.api.models import (
    ReindexRequest,
    ReindexResponse,
)
from agents.policies.agent.api.validators import validate_category
from agents.policies.agent.services.document_service import DocumentService
from agents.policies.agent.services.reindex_service import ReindexService
from libraries.observability.logger import get_console_logger

logger = get_console_logger("policies_api_routes")

router = APIRouter()


@router.post("/kb/reindex", response_model=ReindexResponse)
async def reindex_knowledge_base(
    request: ReindexRequest,
    reindex_service: ReindexService = Depends(get_reindex_service),
):
    logger.info(
        f"Reindex request received: force_rebuild={request.force_rebuild}, "
        f"policy_ids={request.policy_ids}"
    )

    try:
        try:
            request.validate_policy_ids()
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        response = await reindex_service.reindex_documents(request)
        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Reindex operation failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Internal server error during reindex operation"
        )


@router.delete("/kb/clear")
async def clear_index(
    document_service: DocumentService = Depends(get_document_service),
):
    try:
        result = await document_service.clear_all_documents()
        return result
    except Exception as e:
        logger.error(f"Clear index error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Internal server error while clearing index"
        )


@router.get("/kb/status")
async def get_indexing_status(
    reindex_service: ReindexService = Depends(get_reindex_service),
):
    try:
        status = await reindex_service.get_indexing_status()
        return status
    except Exception as e:
        logger.error(f"Get indexing status error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Internal server error while getting status"
        )


@router.post("/kb/upload")
async def upload_document(
    file: UploadFile = File(...),
    category: str = Query("general", description="Document category"),
):
    try:
        validate_category(category)

        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")

        allowed_extensions = {'.pdf', '.docx', '.md', '.txt'}
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
            )

        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty file")

        from agents.policies.ingestion.minio_client import MinIOClient

        async with MinIOClient() as minio_client:
            metadata = await minio_client.upload_to_inbox(
                filename=file.filename,
                content=content,
                mime_type=file.content_type or "application/octet-stream"
            )

        logger.info(f"Uploaded document {file.filename} with ID {metadata.document_id}")

        return {
            "message": "Document uploaded successfully",
            "filename": file.filename,
            "document_id": metadata.document_id,
            "sha256": metadata.sha256,
            "size": metadata.file_size,
            "status": "queued_for_processing"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Internal server error during upload"
        )


@router.get("/kb/upload/status")
async def get_upload_status():
    try:
        from agents.policies.ingestion.minio_client import MinIOClient

        async with MinIOClient() as minio_client:
            inbox_files = await minio_client.list_inbox_files()

            processed_count = 0
            failed_count = 0

        return {
            "inbox_count": len(inbox_files),
            "inbox_files": [
                {
                    "filename": f["filename"],
                    "size": f["size"],
                    "last_modified": f["last_modified"].isoformat() if f["last_modified"] else None,
                    "document_id": f["metadata"].get("document_id")
                }
                for f in inbox_files
            ],
            "processed_count": processed_count,
            "failed_count": failed_count
        }

    except Exception as e:
        logger.error(f"Get upload status error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Internal server error while getting upload status"
        )
