"""
Enums for the EggAI message protocol.

These enums define the valid values for message types, agent names, and other constants.
"""

from enum import Enum


class MessageType(str, Enum):
    """Standard message types in the system."""
    # User messages
    USER_MESSAGE = "user_message"

    # Agent requests (from Triage)
    BILLING_REQUEST = "billing_request"
    CLAIM_REQUEST = "claim_request"
    POLICY_REQUEST = "policy_request"
    ESCALATION_REQUEST = "escalation_request"

    # Agent responses
    AGENT_MESSAGE = "agent_message"
    AGENT_MESSAGE_STREAM_START = "agent_message_stream_start"
    AGENT_MESSAGE_STREAM_WAITING_MESSAGE = "agent_message_stream_waiting_message"
    AGENT_MESSAGE_STREAM_CHUNK = "agent_message_stream_chunk"
    AGENT_MESSAGE_STREAM_END = "agent_message_stream_end"

    # Client-facing (WebSocket) message types
    ASSISTANT_MESSAGE = "assistant_message"
    ASSISTANT_MESSAGE_STREAM_START = "assistant_message_stream_start"
    ASSISTANT_MESSAGE_STREAM_WAITING_MESSAGE = "assistant_message_stream_waiting_message"
    ASSISTANT_MESSAGE_STREAM_CHUNK = "assistant_message_stream_chunk"
    ASSISTANT_MESSAGE_STREAM_END = "assistant_message_stream_end"

    # System messages
    AUDIT_LOG = "audit_log"
    ERROR = "error"


class AgentName(str, Enum):
    """Standard agent names in the system."""
    FRONTEND = "Frontend"
    TRIAGE = "Triage"
    BILLING = "Billing"
    CLAIMS = "Claims"
    POLICIES = "Policies"
    ESCALATION = "Escalation"
    AUDIT = "Audit"


class AuditCategory(str, Enum):
    """Audit event categories."""
    SYSTEM_OPERATIONS = "System Operations"
    USER_COMMUNICATION = "User Communication"
    AGENT_PROCESSING = "Agent Processing"
    ERROR = "Error"
    OTHER = "Other"


class OffsetReset(str, Enum):
    """Kafka consumer offset reset options."""
    LATEST = "latest"      # Start reading at the latest record
    EARLIEST = "earliest"  # Start reading at the earliest record
    NONE = "none"         # Throw exception if no previous offset is found
