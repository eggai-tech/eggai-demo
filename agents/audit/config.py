from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import SettingsConfigDict

from libraries.communication.messaging import MessageType
from libraries.core import BaseAgentConfig

from .types import AuditCategory, AuditConfig

load_dotenv()

MESSAGE_CATEGORIES: dict[str, AuditCategory] = {
    MessageType.AGENT_MESSAGE: "User Communication",
    MessageType.BILLING_REQUEST: "Billing",
    MessageType.POLICY_REQUEST: "Policies",
    MessageType.ESCALATION_REQUEST: "Escalation",
}

class Settings(BaseAgentConfig):
    app_name: str = Field(default="audit_agent")
    prometheus_metrics_port: int = Field(default=9096)

    debug_logging_enabled: bool = Field(default=False)
    audit_channel_name: str = Field(default="audit_logs")
    default_category: AuditCategory = Field(default="Other")

    model_config = SettingsConfigDict(
        env_prefix="AUDIT_", env_file=".env", env_ignore_empty=True, extra="ignore"
    )

settings = Settings()

audit_config = AuditConfig(
    message_categories=MESSAGE_CATEGORIES,
    default_category=settings.default_category,
    enable_debug_logging=settings.debug_logging_enabled,
    audit_channel_name=settings.audit_channel_name,
)
