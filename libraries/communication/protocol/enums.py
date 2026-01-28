from enum import Enum


class MessageType(str, Enum):
    USER_MESSAGE = "user_message"
    BILLING_REQUEST = "billing_request"
    CLAIM_REQUEST = "claim_request"
    POLICY_REQUEST = "policy_request"
    ESCALATION_REQUEST = "escalation_request"
    AGENT_MESSAGE = "agent_message"
    AGENT_MESSAGE_STREAM_START = "agent_message_stream_start"
    AGENT_MESSAGE_STREAM_WAITING_MESSAGE = "agent_message_stream_waiting_message"
    AGENT_MESSAGE_STREAM_CHUNK = "agent_message_stream_chunk"
    AGENT_MESSAGE_STREAM_END = "agent_message_stream_end"
    ASSISTANT_MESSAGE = "assistant_message"
    ASSISTANT_MESSAGE_STREAM_START = "assistant_message_stream_start"
    ASSISTANT_MESSAGE_STREAM_WAITING_MESSAGE = "assistant_message_stream_waiting_message"
    ASSISTANT_MESSAGE_STREAM_CHUNK = "assistant_message_stream_chunk"
    ASSISTANT_MESSAGE_STREAM_END = "assistant_message_stream_end"
    AUDIT_LOG = "audit_log"
    ERROR = "error"


class AgentName(str, Enum):
    FRONTEND = "Frontend"
    TRIAGE = "Triage"
    BILLING = "Billing"
    CLAIMS = "Claims"
    POLICIES = "Policies"
    ESCALATION = "Escalation"
    AUDIT = "Audit"


class AuditCategory(str, Enum):
    SYSTEM_OPERATIONS = "System Operations"
    USER_COMMUNICATION = "User Communication"
    AGENT_PROCESSING = "Agent Processing"
    ERROR = "Error"
    OTHER = "Other"


class OffsetReset(str, Enum):
    LATEST = "latest"
    EARLIEST = "earliest"
    NONE = "none"
