from typing import Any, Literal, TypedDict, TypeGuard

from .enums import AgentName, AuditCategory, MessageType


class MessageData(TypedDict):
    connection_id: str


class ChatMessage(TypedDict):
    role: Literal["user", "assistant", "system"]
    content: str
    agent: str | None


class AgentRequestData(MessageData):
    chat_messages: list[ChatMessage]
    timeout: float | None


class AgentResponseData(MessageData):
    message: str
    agent: str


class StreamChunkData(MessageData):
    chunk: str
    chunk_index: int
    message_id: str


class StreamEndData(MessageData):
    message_id: str
    final_content: str


class AuditEventData(TypedDict):
    message_id: str
    message_type: str
    message_source: str
    channel: str
    category: AuditCategory
    audit_timestamp: str
    content: str | None
    error: dict[str, str] | None


class MessageDict(TypedDict, total=False):
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


class AuditLogMessage(TypedDict):
    id: str
    type: Literal[MessageType.AUDIT_LOG]
    source: Literal[AgentName.AUDIT]
    data: AuditEventData
    traceparent: str | None
    tracestate: str | None


class BillingRequestMessage(TypedDict):
    id: str
    type: Literal[MessageType.BILLING_REQUEST]
    source: str  # Usually AgentName.TRIAGE
    data: AgentRequestData
    traceparent: str | None
    tracestate: str | None


class ClaimRequestMessage(TypedDict):
    id: str
    type: Literal[MessageType.CLAIM_REQUEST]
    source: str  # Usually AgentName.TRIAGE
    data: AgentRequestData
    traceparent: str | None
    tracestate: str | None


class PolicyRequestMessage(TypedDict):
    id: str
    type: Literal[MessageType.POLICY_REQUEST]
    source: str  # Usually AgentName.TRIAGE
    data: AgentRequestData
    traceparent: str | None
    tracestate: str | None


class EscalationRequestMessage(TypedDict):
    id: str
    type: Literal[MessageType.ESCALATION_REQUEST]
    source: str  # Usually AgentName.TRIAGE
    data: AgentRequestData
    traceparent: str | None
    tracestate: str | None


class UserMessage(TypedDict):
    id: str
    type: Literal[MessageType.USER_MESSAGE]
    source: Literal[AgentName.FRONTEND]
    data: AgentRequestData
    traceparent: str | None
    tracestate: str | None


class AgentMessage(TypedDict):
    id: str
    type: Literal[MessageType.AGENT_MESSAGE]
    source: str  # Agent name: AgentName.BILLING, AgentName.CLAIMS, etc.
    data: AgentResponseData
    traceparent: str | None
    tracestate: str | None


class StreamChunkMessage(TypedDict):
    id: str
    type: Literal[MessageType.AGENT_MESSAGE_STREAM_CHUNK]
    source: str  # Agent name
    data: StreamChunkData
    traceparent: str | None
    tracestate: str | None


class StreamEndMessage(TypedDict):
    id: str
    type: Literal[MessageType.AGENT_MESSAGE_STREAM_END]
    source: str  # Agent name
    data: StreamEndData
    traceparent: str | None
    tracestate: str | None


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


def is_audit_log(msg: dict[str, Any]) -> TypeGuard[AuditLogMessage]:
    return (
        msg.get("type") == MessageType.AUDIT_LOG and
        msg.get("source") == AgentName.AUDIT
    )


def is_billing_request(msg: dict[str, Any]) -> TypeGuard[BillingRequestMessage]:
    return msg.get("type") == MessageType.BILLING_REQUEST


def is_claim_request(msg: dict[str, Any]) -> TypeGuard[ClaimRequestMessage]:
    return msg.get("type") == MessageType.CLAIM_REQUEST


def is_policy_request(msg: dict[str, Any]) -> TypeGuard[PolicyRequestMessage]:
    return msg.get("type") == MessageType.POLICY_REQUEST


def is_escalation_request(msg: dict[str, Any]) -> TypeGuard[EscalationRequestMessage]:
    return msg.get("type") == MessageType.ESCALATION_REQUEST


def is_user_message(msg: dict[str, Any]) -> TypeGuard[UserMessage]:
    return (
        msg.get("type") == MessageType.USER_MESSAGE and
        msg.get("source") == AgentName.FRONTEND
    )


def is_agent_message(msg: dict[str, Any]) -> TypeGuard[AgentMessage]:
    return msg.get("type") == MessageType.AGENT_MESSAGE


def is_stream_end(msg: dict[str, Any]) -> TypeGuard[StreamEndMessage]:
    return msg.get("type") == MessageType.AGENT_MESSAGE_STREAM_END
