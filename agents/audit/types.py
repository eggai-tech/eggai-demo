from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

AuditCategory = Literal[
    "User Communication",
    "Billing",
    "Policies",
    "Escalation",
    "Triage",
    "Other",
    "Error",
]

class AuditConfig(BaseModel):
    message_categories: dict[str, AuditCategory] = Field(default_factory=dict)
    default_category: AuditCategory = Field(default="Other")
    enable_debug_logging: bool = Field(default=False)
    audit_channel_name: str = Field(default="audit_logs")

    model_config = {"validate_assignment": True, "extra": "forbid"}

class SecurityContext(BaseModel):
    """Security context for compliance (GKV-ready architecture)."""
    user_id: str | None = Field(default=None)
    tenant_id: str | None = Field(default=None)
    consent_scope: list[str] = Field(default_factory=list)
    retention_policy: str | None = Field(default=None)


class AuditEvent(BaseModel):
    message_id: str = Field(...)
    message_type: str = Field(...)
    message_source: str = Field(...)
    channel: str = Field(...)
    category: AuditCategory = Field(...)
    audit_timestamp: datetime = Field(default_factory=datetime.now)
    content: str | None = Field(default=None)
    error: dict[str, str] | None = Field(default=None)
    security_context: SecurityContext | None = Field(default=None)

    def to_dict(self) -> dict[str, Any]:
        result = self.model_dump()
        result["audit_timestamp"] = self.audit_timestamp.isoformat()
        return result

    model_config = {"validate_assignment": True}
