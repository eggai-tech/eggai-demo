import os
from typing import Any

from temporalio import activity

from libraries.integrations.vector_store import create_vector_store
from libraries.observability.logger import get_console_logger

logger = get_console_logger("ingestion.document_verification")


@activity.defn
async def verify_document_activity(
    file_path: str, index_name: str = "policies_index", force_rebuild: bool = False
) -> dict[str, Any]:
    logger.info(f"Verifying document existence in vector store: {file_path}")

    try:
        if force_rebuild:
            logger.info("Force rebuild enabled, skipping verification check")
            return {
                "success": True,
                "file_exists": False,
                "should_skip": False,
                "force_rebuild": True,
                "reason": "Force rebuild enabled",
            }

        filename = os.path.basename(file_path)
        vector_store = create_vector_store()

        try:
            existing_docs = await vector_store.search_documents(
                query=f"source_file:{filename}",
                max_hits=400,
            )

            if existing_docs:
                logger.info(
                    f"File {filename} exists with {len(existing_docs)} chunks. Recommending skip."
                )
                return {
                    "success": True,
                    "file_exists": True,
                    "should_skip": True,
                    "existing_chunks": len(existing_docs),
                    "existing_doc_ids": [doc["id"] for doc in existing_docs],
                    "reason": f"File {filename} already exists in vector store",
                }
            else:
                logger.info(f"File {filename} not found in vector store, verification passed")
                return {
                    "success": True,
                    "file_exists": False,
                    "should_skip": False,
                    "reason": f"File {filename} not found in vector store",
                }

        except Exception as search_error:
            logger.warning(f"Search failed during verification: {search_error}")
            return {
                "success": True,
                "file_exists": False,
                "should_skip": False,
                "reason": f"Search failed, proceeding with processing: {search_error}",
            }

    except Exception as e:
        logger.error(f"Document verification failed: {e}", exc_info=True)
        return {
            "success": False,
            "file_exists": False,
            "should_skip": False,
            "error_message": str(e),
            "reason": "Verification failed, proceeding with processing",
        }
