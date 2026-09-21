import json
from unittest.mock import AsyncMock
from uuid import uuid4

import jwt
import pytest

from agents.policies.agent import agent as agent_mod
from agents.policies.agent.tools.database.policy_data import get_personal_policy_details
from libraries.communication.messaging import MessageType
from libraries.observability.tracing import TracedMessage
from libraries.security.keycloak import Caller, caller_var


def test_tool_refuses_foreign_policy():
    token = caller_var.set(Caller("jane", "Jane Smith", ["B67890"], []))
    try:
        assert get_personal_policy_details("A12345") == (
            "Not authorised: policy A12345 does not belong to the authenticated customer."
        )
        assert json.loads(get_personal_policy_details("B67890"))["name"] == "Jane Smith"
    finally:
        caller_var.reset(token)


def test_tool_without_caller_returns_any_policy():
    assert json.loads(get_personal_policy_details("A12345"))["name"] == "John Doe"


def test_empty_policy_number_is_not_found_even_with_caller():
    token = caller_var.set(Caller("jane", "Jane Smith", ["B67890"], []))
    try:
        assert get_personal_policy_details("") == "Policy not found."
    finally:
        caller_var.reset(token)


@pytest.mark.asyncio
async def test_handler_refuses_invalid_token(monkeypatch):
    monkeypatch.setattr(agent_mod.keycloak, "url", "http://kc:8080")

    def bad(token):
        raise jwt.InvalidTokenError("wrong audience")

    monkeypatch.setattr(agent_mod.keycloak, "validate", bad)
    error = AsyncMock()
    monkeypatch.setattr("libraries.security.handler.publish_error_message", error)
    process = AsyncMock()
    monkeypatch.setattr(agent_mod, "process_policy_request", process)

    await agent_mod.handle_policy_request(
        TracedMessage(
            id=str(uuid4()),
            type=MessageType.POLICY_REQUEST,
            source="Triage",
            data={
                "chat_messages": [{"role": "user", "content": "premium?"}],
                "connection_id": "conn-1",
                "security_context": {"access_token": "bad"},
            },
        )
    )

    assert "Authentication failed" in error.call_args.kwargs["message"]
    process.assert_not_awaited()


@pytest.mark.asyncio
async def test_handler_sets_caller(monkeypatch):
    monkeypatch.setattr(agent_mod.keycloak, "url", "http://kc:8080")
    monkeypatch.setattr(agent_mod.keycloak, "validate", lambda token: Caller("john", "John Doe", ["A12345"], []))
    seen = {}

    async def process(conversation_string, connection_id, message_id, timeout_seconds=None):
        seen["caller"] = caller_var.get()

    monkeypatch.setattr(agent_mod, "process_policy_request", process)

    await agent_mod.handle_policy_request(
        TracedMessage(
            id=str(uuid4()),
            type=MessageType.POLICY_REQUEST,
            source="Triage",
            data={
                "chat_messages": [{"role": "user", "content": "premium?"}],
                "connection_id": "conn-1",
                "security_context": {"access_token": "good"},
            },
        )
    )

    assert seen["caller"] == Caller("john", "John Doe", ["A12345"], [])
