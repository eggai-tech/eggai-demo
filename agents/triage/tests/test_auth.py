from unittest.mock import AsyncMock
from uuid import uuid4

import jwt
import pytest

from agents.triage import agent as agent_mod
from agents.triage.models import AGENT_REGISTRY, TargetAgent
from libraries.communication.messaging import MessageType
from libraries.observability.tracing import TracedMessage
from libraries.security.keycloak import Caller


def _msg(security_context):
    return TracedMessage(
        id=str(uuid4()),
        type=MessageType.USER_MESSAGE,
        source="Frontend",
        data={
            "chat_messages": [{"role": "user", "content": "How much is my premium?"}],
            "connection_id": "conn-1",
            "security_context": security_context,
        },
    )


def test_registry_maps_agents_to_clients():
    assert AGENT_REGISTRY[TargetAgent.BillingAgent]["client_id"] == "insurance-billing"
    assert AGENT_REGISTRY[TargetAgent.PolicyAgent]["client_id"] == "insurance-policies"
    assert AGENT_REGISTRY[TargetAgent.ClaimsAgent]["client_id"] == "insurance-claims"
    assert AGENT_REGISTRY[TargetAgent.EscalationAgent]["client_id"] == "insurance-escalation"


@pytest.mark.asyncio
async def test_publish_to_agent_exchanges_token(monkeypatch):
    monkeypatch.setattr(agent_mod.keycloak, "url", "http://kc:8080")
    monkeypatch.setattr(agent_mod.keycloak, "exchange", AsyncMock(return_value="billing-token"))
    publish = AsyncMock()
    monkeypatch.setattr(agent_mod.agents_channel, "publish", publish)
    context = {"user_id": "john", "name": "John Doe", "policy_numbers": ["A12345"], "roles": [], "access_token": "triage-token"}

    await agent_mod._publish_to_agent("User: hi\n", TargetAgent.BillingAgent, _msg(context), context)

    agent_mod.keycloak.exchange.assert_awaited_once_with("triage-token", "insurance-billing")
    published = publish.call_args.args[0]
    assert published.data["security_context"]["access_token"] == "billing-token"
    assert published.data["security_context"]["user_id"] == "john"


@pytest.mark.asyncio
async def test_publish_to_agent_without_keycloak_forwards_context(monkeypatch):
    monkeypatch.setattr(agent_mod.keycloak, "url", "")
    publish = AsyncMock()
    monkeypatch.setattr(agent_mod.agents_channel, "publish", publish)
    context = {"user_id": "demo-user-1", "name": "", "policy_numbers": [], "roles": [], "access_token": ""}

    await agent_mod._publish_to_agent("User: hi\n", TargetAgent.PolicyAgent, _msg(context), context)

    assert publish.call_args.args[0].data["security_context"] == context


@pytest.mark.asyncio
async def test_invalid_token_is_refused(monkeypatch):
    monkeypatch.setattr(agent_mod.keycloak, "url", "http://kc:8080")

    def bad(token):
        raise jwt.InvalidTokenError("bad signature")

    monkeypatch.setattr(agent_mod.keycloak, "validate", bad)
    publish = AsyncMock()
    monkeypatch.setattr(agent_mod.human_channel, "publish", publish)
    route = AsyncMock()
    monkeypatch.setattr(agent_mod, "_publish_to_agent", route)

    await agent_mod.handle_user_message(_msg({"access_token": "x"}))

    assert "Authentication failed" in publish.call_args.args[0].data["message"]
    route.assert_not_awaited()


@pytest.mark.asyncio
async def test_identity_line_prefixes_conversation(monkeypatch):
    monkeypatch.setattr(agent_mod.keycloak, "url", "http://kc:8080")
    monkeypatch.setattr(agent_mod.keycloak, "validate", lambda token: Caller("john", "John Doe", ["A12345"], []))
    monkeypatch.setattr(agent_mod, "publish_waiting_message", AsyncMock())
    route = AsyncMock()
    monkeypatch.setattr(agent_mod, "_publish_to_agent", route)

    class Result:
        target_agent = TargetAgent.BillingAgent
        latency_ms = 1.0

    monkeypatch.setattr(agent_mod.current_classifier, "classify", lambda chat_history: Result())
    context = {"user_id": "john", "name": "Mallory", "policy_numbers": ["A12345"], "roles": [], "access_token": "t"}

    await agent_mod.handle_user_message(_msg(context))

    conversation = route.call_args.args[0]
    assert conversation.startswith("Authenticated customer: John Doe, policies: A12345\n")
