from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from dotenv import load_dotenv

from agents.billing.agent import (
    get_conversation_string,
    handle_billing_request,
    handle_other_messages,
    process_billing_request,
)
from libraries.communication.messaging import MessageType
from libraries.observability.tracing import TracedMessage


@pytest.mark.asyncio
async def test_billing_agent_error_handling(monkeypatch):
    load_dotenv()

    async def mock_billing_error(*args, **kwargs):
        # This needs to be an async generator, not a coroutine
        raise Exception("Billing module error")
        yield  # Make this an async generator (unreachable, but defines the function type)

    monkeypatch.setattr(
        "agents.billing.utils.process_billing", mock_billing_error
    )

    from agents.billing.agent import human_stream_channel

    mock_publish = AsyncMock()
    monkeypatch.setattr(human_stream_channel, "publish", mock_publish)

    test_message = TracedMessage(
        id=str(uuid4()),
        type=MessageType.BILLING_REQUEST,
        source="TestAgent",
        data={
            "chat_messages": [{"role": "user", "content": "What's my bill?"}],
            "connection_id": str(uuid4()),
            "message_id": str(uuid4()),
        },
    )

    await handle_billing_request(test_message)

    assert mock_publish.called
    error_calls = [
        call
        for call in mock_publish.call_args_list
        if call[0][0].type == "agent_message_stream_end"
        and "error" in call[0][0].data.get("message", "").lower()
    ]
    assert len(error_calls) > 0


@pytest.mark.asyncio
async def test_billing_empty_chat_messages(monkeypatch):
    load_dotenv()

    from agents.billing.agent import human_channel

    mock_publish = AsyncMock()
    monkeypatch.setattr(human_channel, "publish", mock_publish)

    test_message = TracedMessage(
        id=str(uuid4()),
        type=MessageType.BILLING_REQUEST,
        source="TestAgent",
        data={
            "chat_messages": [],
            "connection_id": str(uuid4()),
            "message_id": str(uuid4()),
        },
    )

    await handle_billing_request(test_message)

    assert mock_publish.called
    error_calls = [
        call
        for call in mock_publish.call_args_list
        if call[0][0].type == "agent_message"
        and "didn't receive any message content" in call[0][0].data.get("message", "")
    ]
    assert len(error_calls) > 0


@pytest.mark.asyncio
async def test_billing_missing_connection_id(monkeypatch):
    load_dotenv()

    from agents.billing.agent import human_stream_channel

    mock_publish = AsyncMock()
    monkeypatch.setattr(human_stream_channel, "publish", mock_publish)

    test_message = TracedMessage(
        id=str(uuid4()),
        type=MessageType.BILLING_REQUEST,
        source="TestAgent",
        data={
            "chat_messages": [{"role": "user", "content": "test"}],
            "message_id": str(uuid4()),
        },
    )

    await handle_billing_request(test_message)

    assert mock_publish.called
    for call in mock_publish.call_args_list:
        msg = call[0][0]
        assert msg.data.get("connection_id") == "unknown"


@pytest.mark.asyncio
async def test_handle_other_messages():
    test_message = TracedMessage(
        id=str(uuid4()),
        type="debug_message",
        source="TestAgent",
        data={"content": "debug info"},
    )

    await handle_other_messages(test_message)


def test_get_conversation_string_empty():
    result = get_conversation_string([])
    assert result == ""


def test_get_conversation_string_missing_content():
    messages = [{"role": "user"}, {"role": "assistant", "content": "Hello"}]
    result = get_conversation_string(messages)
    assert "user:" not in result.lower()
    assert "assistant: Hello" in result


def test_get_conversation_string_normal():
    messages = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello", "agent": "BillingAgent"},
    ]
    result = get_conversation_string(messages)
    assert "user: Hi" in result
    assert "assistant: Hello" in result


@pytest.mark.asyncio
async def test_process_billing_request_short_conversation():
    connection_id = str(uuid4())
    message_id = str(uuid4())

    with pytest.raises(ValueError, match="too short"):
        await process_billing_request("Hi", connection_id, message_id)


@pytest.mark.asyncio
async def test_billing_missing_data_fields(monkeypatch):
    load_dotenv()

    from agents.billing.agent import human_channel

    mock_publish = AsyncMock()
    monkeypatch.setattr(human_channel, "publish", mock_publish)

    test_message = TracedMessage(
        id=str(uuid4()),
        type=MessageType.BILLING_REQUEST,
        source="TestAgent",
        data={},
    )

    await handle_billing_request(test_message)

    assert mock_publish.called
    error_calls = [
        call
        for call in mock_publish.call_args_list
        if call[0][0].type == "agent_message"
        and "didn't receive any message content" in call[0][0].data.get("message", "")
    ]
    assert len(error_calls) > 0


@pytest.mark.asyncio
async def test_billing_exception_in_handler(monkeypatch):
    load_dotenv()

    from agents.billing.agent import human_channel

    mock_publish = AsyncMock()
    monkeypatch.setattr(human_channel, "publish", mock_publish)

    def mock_error(*args, **kwargs):
        raise Exception("Unexpected error")

    monkeypatch.setattr("agents.billing.agent.get_conversation_string", mock_error)

    test_message = TracedMessage(
        id=str(uuid4()),
        type="billing_request",
        source="TestAgent",
        data={
            "chat_messages": [{"role": "user", "content": "test"}],
            "connection_id": str(uuid4()),
        },
    )

    await handle_billing_request(test_message)

    assert mock_publish.called
    error_published = any(
        "trouble processing" in call[0][0].data.get("message", "").lower()
        for call in mock_publish.call_args_list
        if call[0][0].type == "agent_message"
    )
    assert error_published


def test_get_conversation_string_special_characters():
    messages = [
        {"role": "user", "content": "What's my bill? It costs $100.50!"},
        {"role": "assistant", "content": "I'll help you with that. Let me check..."},
    ]
    result = get_conversation_string(messages)
    assert "What's my bill? It costs $100.50!" in result
    assert "I'll help you with that. Let me check..." in result
    assert result.endswith("\n")


def test_get_conversation_string_role_formatting():
    messages = [
        {"role": "User", "content": "Hello"},
        {"role": "BillingAgent", "content": "Hi there"},
        {"role": "user", "content": "Thanks"},
    ]
    result = get_conversation_string(messages)
    assert "User: Hello" in result
    assert "BillingAgent: Hi there" in result
    assert "user: Thanks" in result
