from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import SettingsConfigDict

from libraries.communication.messaging import AgentName
from libraries.core import BaseAgentConfig

from .types import ModelConfig

load_dotenv()

AGENT_NAME = AgentName.ESCALATION
GROUP_ID = "escalation_agent_group"


class Settings(BaseAgentConfig):
    app_name: str = Field(default="escalation_agent")
    prometheus_metrics_port: int = Field(default=9094, description="Port for Prometheus metrics server")

    ticket_database_path: str = Field(default="")
    default_departments: list[str] = Field(default=["Technical Support", "Billing", "Sales"])

    model_name: str = Field(default="ticketing_react", description="Name of the model")
    max_iterations: int = Field(default=5, ge=1, le=10)
    use_tracing: bool = Field(default=True)
    cache_enabled: bool = Field(default=False)
    timeout_seconds: float = Field(default=30.0, description="Timeout for model inference in seconds")
    truncation_length: int = Field(default=15000, ge=1000)

    model_config = SettingsConfigDict(
        env_prefix="ESCALATION_", env_file=".env", env_ignore_empty=True, extra="ignore"
    )


settings = Settings()

model_config = ModelConfig(
    name=settings.model_name,
    max_iterations=settings.max_iterations,
    use_tracing=settings.use_tracing,
    cache_enabled=settings.cache_enabled,
    timeout_seconds=settings.timeout_seconds,
    truncation_length=settings.truncation_length,
)
