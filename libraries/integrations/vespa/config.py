import os

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class VespaConfig(BaseSettings):
    deployment_namespace: str | None = Field(default=None)
    vespa_url: str = Field(default="http://localhost:8080")
    vespa_app_name_base: str = Field(default="policies")
    vespa_timeout: int = Field(default=120)
    vespa_connections: int = Field(default=5)
    schema_name: str = Field(default="policy_document")
    batch_size: int = Field(default=100)
    max_retries: int = Field(default=3)
    max_hits: int = Field(default=10)
    ranking_profile: str = Field(default="default")

    model_config = SettingsConfigDict(
        env_prefix="VESPA_", env_file=".env", env_ignore_empty=True, extra="ignore"
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not self.deployment_namespace:
            self.deployment_namespace = os.getenv("DEPLOYMENT_NAMESPACE")

    @property
    def vespa_app_name(self) -> str:
        if self.deployment_namespace:
            return f"{self.deployment_namespace}-{self.vespa_app_name_base}"
        return self.vespa_app_name_base


vespa_config = VespaConfig()
