import asyncio
import os
import signal
import sys
from pathlib import Path

from temporalio.client import Client

from agents.policies.ingestion.config import settings
from agents.policies.ingestion.temporal_client import TemporalClient
from agents.policies.ingestion.workflows.worker import (
    run_policy_documentation_worker,
)
from agents.policies.vespa.deploy_package import deploy_to_vespa
from libraries.observability.logger import get_console_logger
from libraries.observability.tracing import init_telemetry

logger = get_console_logger("ingestion.start_worker")


async def initialize_minio_and_migrate():
    logger.info("Initializing MinIO and checking for migration needs...")

    try:
        import hashlib

        from agents.policies.ingestion.minio_client import MinIOClient
        from libraries.integrations.vespa import VespaClient

        async with MinIOClient() as minio_client:
            await minio_client.initialize_buckets()
            logger.info("MinIO buckets initialized")

            processed_files = await minio_client.list_processed_files()

            if not processed_files:
                logger.info("No files in MinIO processed folder, running migration...")

                vespa_client = VespaClient()

                existing_docs = await vespa_client.search_documents(
                    query="",
                    max_hits=400,
                )

                logger.info(f"Found {len(existing_docs)} documents in Vespa")

                documents = {}
                for doc in existing_docs:
                    source_file = doc.get('source_file')
                    document_id = doc.get('document_id')

                    if source_file and source_file not in documents:
                        documents[source_file] = document_id

                logger.info(f"Found {len(documents)} unique source files to migrate: {list(documents.keys())}")

                current_dir = Path(__file__).parent
                documents_dir = current_dir / "documents"
                migrated = 0

                for source_file, doc_id in documents.items():
                    try:
                        file_path = documents_dir / source_file
                        logger.debug(f"Checking file path: {file_path}")

                        if file_path.exists():
                            content = file_path.read_bytes()
                            file_hash = hashlib.sha256(content).hexdigest()
                            logger.info(f"Uploading {source_file} (size: {len(content)} bytes, hash: {file_hash[:8]}...)")

                            async with minio_client._get_client() as s3:
                                await s3.put_object(
                                    Bucket=minio_client.bucket_name,
                                    Key=f"processed/{source_file}",
                                    Body=content,
                                    Metadata={
                                        'sha256': file_hash,
                                        'document_id': doc_id,
                                        'original_filename': source_file,
                                        'migrated': 'true'
                                    }
                                )
                            migrated += 1
                            logger.info(f"✓ Migrated {source_file} to MinIO processed folder")
                        else:
                            logger.warning(f"File not found at {file_path}")
                    except Exception as e:
                        logger.error(f"Failed to migrate {source_file}: {e}", exc_info=True)

                logger.info(f"Migration complete: {migrated} documents migrated")
            else:
                logger.info("MinIO already contains processed files, skipping migration")

    except Exception as e:
        logger.error(f"Error during MinIO initialization: {e}", exc_info=True)
        logger.warning("Continuing without MinIO support")


async def start_minio_watcher(client):
    try:
        workflow_id = "minio-inbox-watcher"
        poll_interval = int(os.getenv("MINIO_POLL_INTERVAL", "30"))

        try:
            handle = client.get_workflow_handle(workflow_id)
            desc = await handle.describe()
            if desc.status == 1:  # WorkflowExecutionStatus.RUNNING
                logger.info("MinIO watcher workflow already running")
                return
            else:
                logger.info(f"MinIO watcher workflow exists but not running (status: {desc.status})")
        except Exception:
            logger.info("MinIO watcher workflow not found")

        logger.info("Starting MinIO inbox watcher workflow...")

        from agents.policies.ingestion.workflows.minio_watcher_workflow import (
            MinIOInboxWatcherWorkflow,
        )

        handle = await client.start_workflow(
            MinIOInboxWatcherWorkflow.run,
            args=[poll_interval],
            id=workflow_id,
            task_queue=settings.temporal_task_queue
        )

        logger.info(f"MinIO watcher started with {poll_interval}s interval")

    except Exception as e:
        logger.error(f"Error starting MinIO watcher: {e}", exc_info=True)
        logger.warning("Continuing without MinIO watcher")


async def trigger_initial_document_ingestion():
    logger.info("Starting initial document ingestion for all 4 policies...")

    policy_ids = ["auto", "home", "health", "life"]

    current_dir = Path(__file__).parent
    documents_dir = current_dir / "documents"

    try:
        client = TemporalClient(
            temporal_server_url=settings.temporal_server_url,
            temporal_namespace=settings.get_temporal_namespace(),
            temporal_task_queue=settings.temporal_task_queue,
        )

        total_processed = 0
        total_indexed = 0

        for policy_id in policy_ids:
            policy_file = documents_dir / f"{policy_id}.md"

            if not policy_file.exists():
                logger.warning(f"Policy file not found: {policy_file}")
                continue

            logger.info(f"Processing policy file: {policy_file}")

            result = await client.ingest_document_async(
                file_path=str(policy_file),
                category=policy_id,
                index_name="policies_index",
                force_rebuild=False,
            )

            if result.success:
                if result.skipped:
                    logger.info(f"Policy {policy_id} skipped: {result.skip_reason}")
                else:
                    logger.info(f"Policy {policy_id} ingested successfully!")
                    logger.info(f"  Chunks indexed: {result.total_documents_indexed}")

                total_processed += result.documents_processed
                total_indexed += result.total_documents_indexed
            else:
                logger.error(
                    f"Policy {policy_id} ingestion failed: {result.error_message}"
                )

        logger.info("Initial document ingestion completed!")
        logger.info(f"Total files processed: {total_processed}")
        logger.info(f"Total chunks indexed: {total_indexed}")

        await client.close()

    except Exception as e:
        logger.error(f"Error during initial document ingestion: {e}", exc_info=True)


async def main():
    init_telemetry(app_name=settings.app_name, endpoint=settings.otel_endpoint)

    logger.info("Starting Policy Documentation Temporal worker with settings:")
    logger.info(f"  Server URL: {settings.temporal_server_url}")
    logger.info(f"  Namespace: {settings.get_temporal_namespace()}")
    logger.info(f"  Task Queue: {settings.temporal_task_queue}")

    shutdown_event = asyncio.Event()

    def signal_handler(signum):
        logger.info(f"Received signal {signum}, shutting down...")
        shutdown_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda s=sig: signal_handler(s))

    worker = None
    try:
        worker = await run_policy_documentation_worker()

        logger.info("Policy Documentation worker is running. Press Ctrl+C to stop.")

        logger.info("Ensuring Vespa schema is deployed...")

        schema_deployed = deploy_to_vespa(
            config_server_url=settings.vespa_config_url,
            query_url=settings.vespa_query_url,
            force=True,
            artifacts_dir=settings.vespa_artifacts_dir,
            deployment_mode=settings.vespa_deployment_mode,
            node_count=settings.vespa_node_count,
            hosts_config=settings.vespa_hosts_config,
            services_xml=settings.vespa_services_xml,
            app_name=settings.vespa_app_name,
        )

        if not schema_deployed:
            logger.error(
                "Vespa schema deployment failed - cannot proceed with document ingestion"
            )
            logger.error("Please check Vespa container status and try again")
            raise Exception("Vespa schema deployment failed")

        logger.info("Vespa schema ready - proceeding with document ingestion")

        await trigger_initial_document_ingestion()

        try:
            await initialize_minio_and_migrate()

            temporal_client = await Client.connect(
                settings.temporal_server_url,
                namespace=settings.get_temporal_namespace()
            )
            await start_minio_watcher(temporal_client)
        except Exception as e:
            logger.error(f"MinIO initialization failed: {e}")
            logger.info("Continuing without MinIO support")

        await shutdown_event.wait()

    except KeyboardInterrupt:
        logger.info("Worker shutdown requested by user")
    except Exception as e:
        logger.error(f"Error running Policy Documentation worker: {e}", exc_info=True)
        sys.exit(1)
    finally:
        if worker:
            logger.info("Shutting down worker...")
            try:
                await worker.shutdown()
            except Exception as e:
                logger.error(f"Error during worker shutdown: {e}")
        logger.info("Policy Documentation worker shutdown complete")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        sys.exit(0)
