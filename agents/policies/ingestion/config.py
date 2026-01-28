import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()

class Settings(BaseSettings):
    app_name: str = Field(default="policies_document_ingestion")

    deployment_namespace: str | None = Field(default=None)

    temporal_server_url: str = Field(default="localhost:7233")
    temporal_namespace: str | None = Field(default=None)
    temporal_task_queue_base: str = Field(default="policy-rag")

    otel_endpoint: str = Field(default="http://localhost:4318")

    vespa_config_url: str = Field(default="http://localhost:19071")
    vespa_query_url: str = Field(default="http://localhost:8080")
    vespa_app_name_base: str = Field(default="policies")

    vespa_deployment_mode: str = Field(default="production")
    vespa_node_count: int = Field(default=3)
    vespa_artifacts_dir: Path | None = Field(default=None)
    vespa_hosts_config: Path | None = Field(default=None)
    vespa_services_xml: Path | None = Field(default=None)

    model_config = SettingsConfigDict(
        env_prefix="POLICIES_DOCUMENT_INGESTION_",
        env_file=".env",
        env_ignore_empty=True,
        extra="ignore",
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not self.deployment_namespace:
            self.deployment_namespace = os.getenv("DEPLOYMENT_NAMESPACE")

    def get_temporal_namespace(self) -> str:
        if self.temporal_namespace:
            return self.temporal_namespace
        if self.deployment_namespace:
            return self.deployment_namespace
        return "default"

    @property
    def temporal_task_queue(self) -> str:
        if self.deployment_namespace:
            return f"{self.deployment_namespace}-{self.temporal_task_queue_base}"
        return self.temporal_task_queue_base

    @property
    def vespa_app_name(self) -> str:
        if self.deployment_namespace:
            return f"{self.deployment_namespace}-{self.vespa_app_name_base}"
        return self.vespa_app_name_base


settings = Settings()
