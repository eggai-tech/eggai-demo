from unittest.mock import AsyncMock

import httpx
import jwt
import pytest
from fastapi.testclient import TestClient

from agents.frontend import agent as agent_mod
from agents.frontend import main as main_mod
from libraries.security.keycloak import Caller


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setattr(agent_mod.keycloak, "url", "http://kc:8080")
    monkeypatch.setattr(agent_mod.keycloak, "exchange", AsyncMock(return_value="obo-token"))
    return agent_mod.keycloak


@pytest.mark.asyncio
async def test_security_context_without_keycloak(monkeypatch):
    monkeypatch.setattr(agent_mod.keycloak, "url", "")
    assert await agent_mod.security_context("abcdefgh-1234", None) == {
        "user_id": "demo-user-abcdefgh",
        "name": "",
        "policy_numbers": [],
        "roles": [],
        "access_token": "",
    }


@pytest.mark.asyncio
async def test_security_context_exchanges_for_triage(enabled, monkeypatch):
    monkeypatch.setattr(enabled, "validate", lambda token: Caller("john", "John Doe", ["A12345"], []))
    ctx = await agent_mod.security_context("conn", "browser-token")
    assert ctx["user_id"] == "john"
    assert ctx["access_token"] == "obo-token"
    enabled.exchange.assert_awaited_once_with("browser-token", "insurance-triage")


@pytest.mark.asyncio
async def test_security_context_rejects_invalid_token(enabled, monkeypatch):
    def bad(token):
        raise jwt.InvalidTokenError("expired")

    monkeypatch.setattr(enabled, "validate", bad)
    with pytest.raises(jwt.InvalidTokenError):
        await agent_mod.security_context("conn", "browser-token")


def test_config_reports_keycloak(monkeypatch):
    monkeypatch.setattr(main_mod.settings, "keycloak_url", "http://kc:8080")
    monkeypatch.setattr(main_mod.settings, "keycloak_public_url", "http://localhost:8180")
    body = TestClient(main_mod.api).get("/config").json()
    assert body["keycloak"] == {"url": "http://localhost:8180", "realm": "insurance", "clientId": "insurance-web"}


def test_config_without_keycloak(monkeypatch):
    monkeypatch.setattr(main_mod.settings, "keycloak_url", "")
    assert TestClient(main_mod.api).get("/config").json()["keycloak"] is None


def test_proxy_requires_bearer(enabled):
    response = TestClient(main_mod.api).get("/api/policies/policies")
    assert response.status_code == 401


def test_proxy_requires_admin_role(enabled, monkeypatch):
    monkeypatch.setattr(enabled, "validate", lambda token: Caller("john", "John Doe", ["A12345"], []))
    response = TestClient(main_mod.api).get("/api/policies/policies", headers={"Authorization": "Bearer t"})
    assert response.status_code == 403


def test_proxy_forwards_exchanged_token(enabled, monkeypatch):
    monkeypatch.setattr(enabled, "validate", lambda token: Caller("alice", "Alice", ["C24680"], ["insurance-admin"]))
    upstream = AsyncMock(return_value=httpx.Response(200, json={"ok": True}))
    monkeypatch.setattr(main_mod.upstream, "request", upstream)

    response = TestClient(main_mod.api).get("/api/billing/billing", headers={"Authorization": "Bearer t"})

    assert response.status_code == 200
    enabled.exchange.assert_awaited_once_with("t", "insurance-billing")
    assert upstream.call_args.kwargs["headers"]["authorization"] == "Bearer obo-token"
