from libraries.observability.logger import get_console_logger

from .base import VectorStoreBase
from .config import VectorStoreBackend, VectorStoreConfig, vector_store_config

logger = get_console_logger("vector_store.factory")


def create_vector_store(config: VectorStoreConfig | None = None) -> VectorStoreBase:
    """Create a vector store instance based on configuration.

    Set VECTOR_STORE_BACKEND=faiss (default) or VECTOR_STORE_BACKEND=vespa in .env
    """
    cfg = config or vector_store_config

    if cfg.backend == VectorStoreBackend.VESPA:
        logger.info("Using Vespa vector store backend")
        from .vespa_adapter import VespaVectorStore

        return VespaVectorStore()
    else:
        logger.info("Using in-memory FAISS vector store backend")
        from .store import InMemoryVectorStore

        return InMemoryVectorStore(config=cfg)
