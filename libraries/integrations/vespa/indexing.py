from asyncio import Semaphore, gather
from typing import Any

import httpx
from tenacity import RetryError, retry, stop_after_attempt, wait_exponential

from libraries.observability.logger import get_console_logger
from libraries.observability.tracing import create_tracer

from .schemas import PolicyDocument

logger = get_console_logger("vespa_client")
tracer = create_tracer("vespa", "client")


class VespaIndexingMixin:
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2))
    async def _upload_single_document(self, session, document: PolicyDocument) -> bool:
        logger.debug(f"Uploading document: {document.id}")

        try:
            response = await session.feed_data_point(
                data_id=document.id,
                fields=document.to_vespa_dict(),
                schema=self.config.schema_name,
            )

            if not response.is_successful():
                logger.error(
                    f"Failed to upload document {document.id}: "
                    f"Status: {response.status_code}, Content: {response.json}"
                )
                raise Exception(f"Upload failed for document {document.id}")

            logger.debug(f"Successfully uploaded document {document.id}")
            return True

        except Exception as e:
            logger.error(f"Error uploading document {document.id}: {e}")
            raise

    @tracer.start_as_current_span("index_documents")
    async def index_documents(self, documents: list[PolicyDocument]) -> dict[str, Any]:
        logger.info(f"Starting indexing of {len(documents)} documents")

        if not await self.check_connectivity():
            raise Exception("Cannot connect to Vespa")

        success_count = 0
        error_count = 0
        errors = []

        async with self.vespa_app.asyncio(
            connections=self.config.vespa_connections,
            timeout=httpx.Timeout(self.config.vespa_timeout),
        ) as session:
            semaphore = Semaphore(self.config.vespa_connections)

            async def upload_with_limit(doc: PolicyDocument):
                nonlocal success_count, error_count
                async with semaphore:
                    try:
                        await self._upload_single_document(session, doc)
                        success_count += 1
                    except RetryError as e:
                        error_count += 1
                        error_msg = f"Final failure for document {doc.id}: {e}"
                        logger.error(error_msg)
                        errors.append(error_msg)
                    except Exception as e:
                        error_count += 1
                        error_msg = f"Unexpected error for document {doc.id}: {e}"
                        logger.error(error_msg)
                        errors.append(error_msg)

            tasks = [upload_with_limit(doc) for doc in documents]
            await gather(*tasks, return_exceptions=True)

        result = {
            "total_documents": len(documents),
            "successful": success_count,
            "failed": error_count,
            "errors": errors,
        }

        logger.info(
            f"Indexing completed: {success_count} successful, {error_count} failed"
        )

        return result
