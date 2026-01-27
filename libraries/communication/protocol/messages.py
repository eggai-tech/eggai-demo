"""
Message type definitions for the EggAI protocol.

These define the structure of messages exchanged between agents.
"""

from typing import Any, Literal, TypedDict, TypeGuard

from .enums import AgentName, AuditCategory, MessageType


# Base message data payloads
class MessageData(TypedDict):
    """Base message data payload."""
    connection_id: str


class ChatMessage(TypedDict):
    """Individual chat message structure."""
    role: Literal["user", "assistant", "system"]
    content: str
    agent: str | None


class AgentRequestData(MessageData):
    """Data payload for agent request messages."""
    chat_messages: list[ChatMessage]
    timeout: float | None


class AgentResponseData(MessageData):
    """Data payload for agent response messages."""
    message: str
    agent: str


class StreamChunkData(MessageData):
    """Data payload for streaming chunk messages."""
    chunk: str
    chunk_index: int
    message_id: str


class StreamEndData(MessageData):
    """Data payload for stream end messages."""
    message_id: str
    final_content: str


class AuditEventData(TypedDict):
    """Data payload for audit events."""
    message_id: str
    message_type: str
    message_source: str
    channel: str
    category: AuditCategory
    audit_timestamp: str
    content: str | None
    error: dict[str, str] | None


# Base message structure
class MessageDict(TypedDict, total=False):
    """Base message dictionary structure."""
    id: str
    type: str
    source: str
    data: dict[str, Any]
    specversion: str
    datacontenttype: str
    subject: str | None
    time: str | None
    traceparent: str | None
    tracestate: str | None
    service_tier: str | None


# Specific message types for each domain
class AuditLogMessage(TypedDict):
    """Audit log message structure."""
    id: str
    type: Literal[MessageType.AUDIT_LOG]
    source: Literal[AgentName.AUDIT]
    data: AuditEventData
    traceparent: str | None
    tracestate: str | None


class BillingRequestMessage(TypedDict):
    """Billing request message structure."""
    id: str
    type: Literal[MessageType.BILLING_REQUEST]
    source: str  # Usually AgentName.TRIAGE
    data: AgentRequestData
    traceparent: str | None
    tracestate: str | None


class ClaimRequestMessage(TypedDict):
    """Claim request message structure."""
    id: str
    type: Literal[MessageType.CLAIM_REQUEST]
    source: str  # Usually AgentName.TRIAGE
    data: AgentRequestData
    traceparent: str | None
    tracestate: str | None


class PolicyRequestMessage(TypedDict):
    """Policy request message structure."""
    id: str
    type: Literal[MessageType.POLICY_REQUEST]
    source: str  # Usually AgentName.TRIAGE
    data: AgentRequestData
    traceparent: str | None
    tracestate: str | None


class EscalationRequestMessage(TypedDict):
    """Escalation request message structure."""
    id: str
    type: Literal[MessageType.ESCALATION_REQUEST]
    source: str  # Usually AgentName.TRIAGE
    data: AgentRequestData
    traceparent: str | None
    tracestate: str | None


class UserMessage(TypedDict):
    """User message from frontend."""
    id: str
    type: Literal[MessageType.USER_MESSAGE]
    source: Literal[AgentName.FRONTEND]
    data: AgentRequestData
    traceparent: str | None
    tracestate: str | None


class AgentMessage(TypedDict):
    """Agent response message."""
    id: str
    type: Literal[MessageType.AGENT_MESSAGE]
    source: str  # Agent name: AgentName.BILLING, AgentName.CLAIMS, etc.
    data: AgentResponseData
    traceparent: str | None
    tracestate: str | None


class StreamChunkMessage(TypedDict):
    """Streaming chunk message."""
    id: str
    type: Literal[MessageType.AGENT_MESSAGE_STREAM_CHUNK]
    source: str  # Agent name
    data: StreamChunkData
    traceparent: str | None
    tracestate: str | None


class StreamEndMessage(TypedDict):
    """Stream end message."""
    id: str
    type: Literal[MessageType.AGENT_MESSAGE_STREAM_END]
    source: str  # Agent name
    data: StreamEndData
    traceparent: str | None
    tracestate: str | None


# Union of all message types
SpecificMessage = (
    AuditLogMessage
    | BillingRequestMessage
    | ClaimRequestMessage
    | PolicyRequestMessage
    | EscalationRequestMessage
    | UserMessage
    | AgentMessage
    | StreamChunkMessage
    | StreamEndMessage
)


# Type guards
def is_audit_log(msg: dict[str, Any]) -> TypeGuard[AuditLogMessage]:
    """Type guard for audit log messages."""
    return (
        msg.get("type") == MessageType.AUDIT_LOG and
        msg.get("source") == AgentName.AUDIT
    )


def is_billing_request(msg: dict[str, Any]) -> TypeGuard[BillingRequestMessage]:
    """Type guard for billing request messages."""
    return msg.get("type") == MessageType.BILLING_REQUEST


def is_claim_request(msg: dict[str, Any]) -> TypeGuard[ClaimRequestMessage]:
    """Type guard for claim request messages."""
    return msg.get("type") == MessageType.CLAIM_REQUEST


def is_policy_request(msg: dict[str, Any]) -> TypeGuard[PolicyRequestMessage]:
    """Type guard for policy request messages."""
    return msg.get("type") == MessageType.POLICY_REQUEST


def is_escalation_request(msg: dict[str, Any]) -> TypeGuard[EscalationRequestMessage]:
    """Type guard for escalation request messages."""
    return msg.get("type") == MessageType.ESCALATION_REQUEST


def is_user_message(msg: dict[str, Any]) -> TypeGuard[UserMessage]:
    """Type guard for user messages."""
    return (
        msg.get("type") == MessageType.USER_MESSAGE and
        msg.get("source") == AgentName.FRONTEND
    )


def is_agent_message(msg: dict[str, Any]) -> TypeGuard[AgentMessage]:
    """Type guard for agent response messages."""
    return msg.get("type") == MessageType.AGENT_MESSAGE


def is_stream_end(msg: dict[str, Any]) -> TypeGuard[StreamEndMessage]:
    """Type guard for stream end messages."""
    return msg.get("type") == MessageType.AGENT_MESSAGE_STREAM_END
