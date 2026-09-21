import json
from unittest.mock import AsyncMock
from uuid import uuid4

import jwt
import pytest

from agents.escalation import agent as agent_mod
from agents.escalation.dspy_modules.escalation import create_ticket, get_tickets_by_policy
from libraries.communication.messaging import MessageType
from libraries.observability.tracing import TracedMessage
from libraries.security.keycloak import Caller, caller_var

REFUSAL = "Not authorised: policy A12345 does not belong to the authenticated customer."


def test_ticket_tools_refuse_foreign_policy():
    token = caller_var.set(Caller("jane", "Jane Smith", ["B67890"], []))
    try:
        assert json.loads(get_tickets_by_policy("A12345")) == {"found": False, "message": REFUSAL, "tickets": []}
        assert json.loads(create_ticket("A12345", "Technical Support", "Login broken", "jane@example.com")) == {"error": REFUSAL}
        assert json.loads(get_tickets_by_policy("B67890"))["found"] is False
    finally:
        caller_var.reset(token)


def test_ticket_tools_without_caller():
    assert json.loads(get_tickets_by_policy("A12345"))["found"] is True


def _msg(token):
    return TracedMessage(
        id=str(uuid4()),
        type=MessageType.ESCALATION_REQUEST,
        source="Triage",
        data={
            "chat_messages": [{"role": "user", "content": "I need a manager"}],
            "connection_id": "conn-1",
            "security_context": {"access_token": token},
        },
    )


@pytest.mark.asyncio
async def test_handler_refuses_invalid_token(monkeypatch):
    monkeypatch.setattr(agent_mod.keycloak, "url", "http://kc:8080")

    def bad(token):
        raise jwt.InvalidTokenError("expired")

    monkeypatch.setattr(agent_mod.keycloak, "validate", bad)
    error = AsyncMock()
    monkeypatch.setattr("libraries.security.handler.publish_error_message", error)
    process = AsyncMock()
    monkeypatch.setattr(agent_mod, "process_escalation_request", process)

    await agent_mod.handle_ticketing_request(_msg("bad"))

    assert "Authentication failed" in error.call_args.kwargs["message"]
    process.assert_not_awaited()


@pytest.mark.asyncio
async def test_handler_sets_caller(monkeypatch):
    monkeypatch.setattr(agent_mod.keycloak, "url", "http://kc:8080")
    monkeypatch.setattr(agent_mod.keycloak, "validate", lambda token: Caller("john", "John Doe", ["A12345"], []))
    seen = {}

    async def process(conversation_string, connection_id, message_id, timeout_seconds=None):
        seen["caller"] = caller_var.get()

    monkeypatch.setattr(agent_mod, "process_escalation_request", process)

    await agent_mod.handle_ticketing_request(_msg("good"))

    assert seen["caller"] == Caller("john", "John Doe", ["A12345"], [])
