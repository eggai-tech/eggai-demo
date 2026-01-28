import os

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ChannelConfig(BaseSettings):
    deployment_namespace: str | None = Field(default=None)
    agents_base: str = Field(default="agents")
    human_base: str = Field(default="human")
    human_stream_base: str = Field(default="human_stream")
    audit_logs_base: str = Field(default="audit_logs")
    metrics_base: str = Field(default="metrics")
    debug_base: str = Field(default="debug")

    model_config = SettingsConfigDict(
        env_prefix="CHANNEL_", env_file=".env", env_ignore_empty=True, extra="ignore"
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not self.deployment_namespace:
            self.deployment_namespace = os.getenv("DEPLOYMENT_NAMESPACE")

    def _apply_namespace(self, base_name: str) -> str:
        if self.deployment_namespace:
            return f"{self.deployment_namespace}-{base_name}"
        return base_name

    @property
    def agents(self) -> str:
        return self._apply_namespace(self.agents_base)

    @property
    def human(self) -> str:
        return self._apply_namespace(self.human_base)

    @property
    def human_stream(self) -> str:
        return self._apply_namespace(self.human_stream_base)

    @property
    def audit_logs(self) -> str:
        return self._apply_namespace(self.audit_logs_base)

    @property
    def metrics(self) -> str:
        return self._apply_namespace(self.metrics_base)

    @property
    def debug(self) -> str:
        return self._apply_namespace(self.debug_base)


channels = ChannelConfig()


async def clear_channels():
    from aiokafka.admin import AIOKafkaAdminClient

    from agents.triage.config import Settings

    settings = Settings()
    admin = AIOKafkaAdminClient(bootstrap_servers=settings.kafka_bootstrap_servers)

    await admin.start()

    try:
        await admin.delete_topics(
            [
                channels.human,
                channels.human_stream,
                channels.agents,
                channels.audit_logs,
            ]
        )
    except Exception as e:
        print(f"Error deleting topics: {e}")
    finally:
        await admin.close()


if __name__ == "__main__":
    import asyncio

    asyncio.run(clear_channels())
