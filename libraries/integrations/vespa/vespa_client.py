import httpx
from vespa.application import Vespa

from libraries.observability.logger import get_console_logger
from libraries.observability.tracing import create_tracer

from .config import VespaConfig
from .indexing import VespaIndexingMixin
from .search import VespaSearchMixin

logger = get_console_logger("vespa_client")
tracer = create_tracer("vespa", "client")


class VespaClientBase:
    def __init__(self, config: VespaConfig | None = None):
        self.config = config or VespaConfig()
        self._vespa_app: Vespa | None = None

    @property
    def vespa_app(self) -> Vespa:
        if self._vespa_app is None:
            self._vespa_app = Vespa(url=self.config.vespa_url)
        return self._vespa_app

    @tracer.start_as_current_span("check_connectivity")
    async def check_connectivity(self) -> bool:
        try:
            async with self.vespa_app.asyncio(
                connections=1, timeout=httpx.Timeout(5.0)
            ) as session:
                await session.query(
                    yql=f"select * from {self.config.schema_name} where true limit 1"
                )
                logger.info("Vespa connectivity check successful")
                return True
        except Exception as e:
            logger.error(f"Vespa connectivity check failed: {e}")
            return False


class VespaClient(VespaClientBase, VespaIndexingMixin, VespaSearchMixin):
    """Vespa client combining base functionality with indexing and search capabilities."""

    pass
