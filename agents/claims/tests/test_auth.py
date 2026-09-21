from unittest.mock import AsyncMock
from uuid import uuid4

import jwt
import pytest

from agents.claims import agent as agent_mod
from agents.claims.dspy_modules.claims_data import file_claim, get_claim_status, update_claim_info
from libraries.communication.messaging import MessageType
from libraries.observability.tracing import TracedMessage
from libraries.security.keycloak import Caller, caller_var

REFUSAL = "Not authorised: policy A12345 does not belong to the authenticated customer."


def test_claims_tools_refuse_foreign_policy():
    token = caller_var.set(Caller("jane", "Jane Smith", ["B67890"], []))
    try:
        assert REFUSAL in get_claim_status("1001")
        assert REFUSAL in update_claim_info("1001", "status", "Closed")
        assert REFUSAL in file_claim("A12345", "Rear-ended at a red light")
        assert '"claim_number": "1002"' in get_claim_status("1002")
    finally:
        caller_var.reset(token)


def test_claims_tools_without_caller():
    assert '"claim_number": "1001"' in get_claim_status("1001")


def _msg(token):
    return TracedMessage(
        id=str(uuid4()),
        type=MessageType.CLAIM_REQUEST,
        source="Triage",
        data={
            "chat_messages": [{"role": "user", "content": "claim status?"}],
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
    monkeypatch.setattr(agent_mod, "process_claims_request", process)

    await agent_mod.handle_claim_request(_msg("bad"))

    assert "Authentication failed" in error.call_args.kwargs["message"]
    process.assert_not_awaited()


@pytest.mark.asyncio
async def test_handler_sets_caller(monkeypatch):
    monkeypatch.setattr(agent_mod.keycloak, "url", "http://kc:8080")
    monkeypatch.setattr(agent_mod.keycloak, "validate", lambda token: Caller("john", "John Doe", ["A12345"], []))
    seen = {}

    async def process(conversation_string, connection_id, message_id, timeout_seconds=None):
        seen["caller"] = caller_var.get()

    monkeypatch.setattr(agent_mod, "process_claims_request", process)

    await agent_mod.handle_claim_request(_msg("good"))

    assert seen["caller"] == Caller("john", "John Doe", ["A12345"], [])
