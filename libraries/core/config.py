from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class BaseAgentConfig(BaseSettings):
    app_name: str = Field(...)
    deployment_namespace: str | None = Field(default=None)
    language_model: str = Field(default="openai/gpt-4o-mini")
    language_model_api_base: str | None = Field(default=None)
    cache_enabled: bool = Field(default=False)
    kafka_bootstrap_servers: str = Field(default="localhost:19092")
    kafka_ca_content: str = Field(default="")
    kafka_topic_prefix: str = Field(default="eggai")
    kafka_rebalance_timeout_ms: int = Field(default=20000)
    otel_endpoint: str = Field(default="http://localhost:4318")
    tracing_enabled: bool = Field(default=True)
    prometheus_metrics_port: int = Field(...)
    temporal_namespace: str | None = Field(default=None)
    temporal_task_queue: str | None = Field(default=None)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_ignore_empty=True,
        extra="ignore"
    )

    def get_temporal_namespace(self) -> str:
        if self.temporal_namespace:
            return self.temporal_namespace
        if self.deployment_namespace:
            return self.deployment_namespace
        return "default"

    def get_temporal_task_queue(self, default_queue: str) -> str:
        base_queue = self.temporal_task_queue or default_queue
        if self.deployment_namespace:
            return f"{self.deployment_namespace}-{base_queue}"
        return base_queue
