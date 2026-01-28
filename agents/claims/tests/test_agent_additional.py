"""Additional tests for claims agent to improve coverage."""

from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from dotenv import load_dotenv

from agents.claims.agent import (
    handle_claim_request,
    handle_other_messages,
    process_claims_request,
)
from libraries.communication.streaming import get_conversation_string
from libraries.observability.tracing import TracedMessage


@pytest.mark.asyncio
async def test_claims_agent_error_handling(monkeypatch):
    load_dotenv()

    def mock_claims_error(*args, **kwargs):
        async def error_generator():
            raise Exception("Claims module error")

        return error_generator()

    monkeypatch.setattr("agents.claims.agent.process_claims", mock_claims_error)

    from agents.claims.agent import human_stream_channel

    mock_publish = AsyncMock()
    monkeypatch.setattr(human_stream_channel, "publish", mock_publish)

    test_message = TracedMessage(
        id=str(uuid4()),
        type="claim_request",
        source="TestAgent",
        data={
            "chat_messages": [{"role": "user", "content": "I need to file a claim"}],
            "connection_id": str(uuid4()),
            "message_id": str(uuid4()),
        },
    )

    await handle_claim_request(test_message)

    assert mock_publish.called
    error_calls = [
        call
        for call in mock_publish.call_args_list
        if call[0][0].type == "agent_message_stream_end"
        and "error" in call[0][0].data.get("message", "").lower()
    ]
    assert len(error_calls) > 0


@pytest.mark.asyncio
async def test_claims_empty_chat_messages(monkeypatch):
    load_dotenv()

    from agents.claims.agent import human_channel

    mock_publish = AsyncMock()
    monkeypatch.setattr(human_channel, "publish", mock_publish)

    test_message = TracedMessage(
        id=str(uuid4()),
        type="claim_request",
        source="TestAgent",
        data={
            "chat_messages": [],
            "connection_id": str(uuid4()),
            "message_id": str(uuid4()),
        },
    )

    await handle_claim_request(test_message)

    assert mock_publish.called
    error_calls = [
        call
        for call in mock_publish.call_args_list
        if call[0][0].type == "agent_message"
        and "didn't receive any message content" in call[0][0].data.get("message", "")
    ]
    assert len(error_calls) > 0


@pytest.mark.asyncio
async def test_claims_missing_connection_id(monkeypatch):
    load_dotenv()

    from agents.claims.agent import human_stream_channel

    mock_publish = AsyncMock()
    monkeypatch.setattr(human_stream_channel, "publish", mock_publish)

    test_message = TracedMessage(
        id=str(uuid4()),
        type="claim_request",
        source="TestAgent",
        data={
            "chat_messages": [{"role": "user", "content": "test"}],
            "message_id": str(uuid4()),
        },
    )

    await handle_claim_request(test_message)

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
    # Messages without content are skipped, only valid messages included
    assert "assistant: Hello" in result


def test_get_conversation_string_normal():
    messages = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
    ]
    result = get_conversation_string(messages)
    assert "user: Hi" in result
    assert "assistant: Hello" in result


def test_claims_specific_functionality():
    from libraries.communication.streaming import get_conversation_string

    messages = [
        {"role": "user", "content": "I need to file a claim"},
        {
            "role": "assistant",
            "content": "I can help with that",
            "agent": "ClaimsAgent",
        },
    ]
    result = get_conversation_string(messages)
    assert "user: I need to file a claim" in result
    assert "assistant: I can help with that" in result


@pytest.mark.asyncio
async def test_claims_exception_in_handler(monkeypatch):
    load_dotenv()

    from agents.claims.agent import human_channel

    mock_publish = AsyncMock()
    monkeypatch.setattr(human_channel, "publish", mock_publish)

    def mock_error(*args, **kwargs):
        raise Exception("Unexpected error")

    monkeypatch.setattr("agents.claims.agent.get_conversation_string", mock_error)

    test_message = TracedMessage(
        id=str(uuid4()),
        type="claim_request",
        source="TestAgent",
        data={
            "chat_messages": [{"role": "user", "content": "test"}],
            "connection_id": str(uuid4()),
        },
    )

    await handle_claim_request(test_message)

    assert mock_publish.called
    error_published = any(
        "trouble processing" in call[0][0].data.get("message", "").lower()
        for call in mock_publish.call_args_list
        if call[0][0].type == "agent_message"
    )
    assert error_published


@pytest.mark.asyncio
async def test_claims_missing_data_fields(monkeypatch):
    load_dotenv()

    from agents.claims.agent import human_channel

    mock_publish = AsyncMock()
    monkeypatch.setattr(human_channel, "publish", mock_publish)

    test_message = TracedMessage(
        id=str(uuid4()),
        type="claim_request",
        source="TestAgent",
        data={},
    )

    await handle_claim_request(test_message)

    assert mock_publish.called


@pytest.mark.asyncio
async def test_process_claims_request_short_conversation():
    connection_id = str(uuid4())
    message_id = str(uuid4())

    with pytest.raises(ValueError, match="too short"):
        await process_claims_request("Hi", connection_id, message_id)


def test_get_conversation_string_special_characters():
    messages = [
        {"role": "user", "content": "My car was damaged! The repair cost is $2,500."},
        {"role": "assistant", "content": "I'll help you file that claim."},
    ]
    result = get_conversation_string(messages)
    assert "My car was damaged! The repair cost is $2,500." in result
    assert "I'll help you file that claim." in result
    assert result.endswith("\n")


def test_get_conversation_string_role_formatting():
    messages = [
        {"role": "User", "content": "Hello"},
        {"role": "ClaimsAgent", "content": "Hi there"},
        {"role": "user", "content": "Thanks"},
    ]
    result = get_conversation_string(messages)
    assert "User: Hello" in result
    assert "ClaimsAgent: Hi there" in result
    assert "user: Thanks" in result
