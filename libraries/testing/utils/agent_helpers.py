import asyncio
from uuid import uuid4

from libraries.communication.protocol import ChatMessage, MessageType
from libraries.observability.tracing import TracedMessage


def create_mock_agent_response(
    agent_name: str,
    message: str,
    connection_id: str = "test-connection-123"
) -> TracedMessage:
    return TracedMessage(
        id=str(uuid4()),
        type=MessageType.AGENT_MESSAGE,
        source=agent_name,
        data={
            "connection_id": connection_id,
            "message": message,
            "agent": agent_name
        }
    )


def create_mock_request_message(
    request_type: MessageType,
    chat_messages: list[ChatMessage],
    connection_id: str = "test-connection-123",
    source: str = "Triage"
) -> TracedMessage:
    return TracedMessage(
        id=str(uuid4()),
        type=request_type,
        source=source,
        data={
            "connection_id": connection_id,
            "chat_messages": chat_messages,
            "timeout": 30.0
        }
    )


async def wait_for_message_with_timeout(
    response_queue: asyncio.Queue,
    timeout: float = 5.0,
    expected_source: str | None = None,
    expected_type: str | None = None
) -> TracedMessage | None:
    try:
        while True:
            message = await asyncio.wait_for(response_queue.get(), timeout=timeout)

            if expected_source and message.source != expected_source:
                continue
            if expected_type and message.type != expected_type:
                continue

            return message

    except TimeoutError:
        return None


def create_mock_audit_log(
    message_id: str,
    message_type: str,
    message_source: str,
    channel: str,
    category: str = "Agent Processing"
) -> TracedMessage:
    return TracedMessage(
        id=str(uuid4()),
        type=MessageType.AUDIT_LOG,
        source="Audit",
        data={
            "message_id": message_id,
            "message_type": message_type,
            "message_source": message_source,
            "channel": channel,
            "category": category,
            "audit_timestamp": "2024-01-01T00:00:00Z",
            "content": None,
            "error": None
        }
    )


def assert_valid_agent_response(
    response: TracedMessage,
    expected_agent: str,
    connection_id: str = "test-connection-123"
) -> None:
    assert response.type == MessageType.AGENT_MESSAGE
    assert response.source == expected_agent
    assert response.data["connection_id"] == connection_id
    assert response.data["agent"] == expected_agent
    assert "message" in response.data
    assert isinstance(response.data["message"], str)
    assert len(response.data["message"]) > 0
