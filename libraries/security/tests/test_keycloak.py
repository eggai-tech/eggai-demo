import time
from urllib.parse import parse_qsl

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

from libraries.security import keycloak as kc


class Settings:
    keycloak_url = "http://kc:8080"
    keycloak_realm = "insurance"
    keycloak_client_id = "insurance-triage"
    keycloak_client_secret = "secret"


SCOPE = "api://insurance-triage/Route.ReadWrite"


@pytest.fixture
def rsa_key():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


@pytest.fixture
def make_token(rsa_key, monkeypatch):
    class SigningKey:
        def __init__(self, key):
            self.key = key

    monkeypatch.setattr(
        kc.PyJWKClient,
        "get_signing_key_from_jwt",
        lambda self, token: SigningKey(rsa_key.public_key()),
    )

    def _make(**overrides):
        claims = {
            "preferred_username": "john",
            "name": "John Doe",
            "aud": "insurance-triage",
            "scp": SCOPE,
            "policy_numbers": ["A12345"],
            "roles": [],
            "exp": int(time.time()) + 300,
            **overrides,
        }
        return jwt.encode(claims, rsa_key, algorithm="RS256", headers={"kid": "test"})

    return _make


def test_validate_returns_caller(make_token):
    caller = kc.Keycloak(Settings(), scope=SCOPE).validate(make_token())
    assert caller == kc.Caller("john", "John Doe", ["A12345"], [])


def test_validate_rejects_wrong_audience(make_token):
    with pytest.raises(jwt.InvalidTokenError):
        kc.Keycloak(Settings(), scope=SCOPE).validate(make_token(aud="insurance-billing"))


def test_validate_rejects_missing_scope(make_token):
    with pytest.raises(jwt.InvalidTokenError):
        kc.Keycloak(Settings(), scope=SCOPE).validate(make_token(scp="api://other/X"))


def test_validate_rejects_expired(make_token):
    with pytest.raises(jwt.InvalidTokenError):
        kc.Keycloak(Settings(), scope=SCOPE).validate(make_token(exp=int(time.time()) - 60))


def test_validate_rejects_empty_token(make_token):
    with pytest.raises(jwt.InvalidTokenError):
        kc.Keycloak(Settings(), scope=SCOPE).validate("")


def test_disabled_without_url():
    class Off(Settings):
        keycloak_url = ""

    assert kc.Keycloak(Off(), scope=SCOPE).enabled is False
    assert kc.Keycloak(Settings(), scope=SCOPE).enabled is True


@pytest.mark.asyncio
async def test_exchange_posts_token_exchange_grant(monkeypatch):
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["form"] = dict(parse_qsl(request.content.decode()))
        return httpx.Response(200, json={"access_token": "obo-token"})

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        kc.httpx,
        "AsyncClient",
        lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw),
    )

    token = await kc.Keycloak(Settings(), scope=SCOPE).exchange("user-token", "insurance-billing")

    assert token == "obo-token"
    assert seen["url"] == "http://kc:8080/realms/insurance/protocol/openid-connect/token"
    assert seen["form"] == {
        "grant_type": "urn:ietf:params:oauth:grant-type:token-exchange",
        "subject_token": "user-token",
        "subject_token_type": "urn:ietf:params:oauth:token-type:access_token",
        "requested_token_type": "urn:ietf:params:oauth:token-type:access_token",
        "audience": "insurance-billing",
        "scope": "insurance-billing-scope",
        "client_id": "insurance-triage",
        "client_secret": "secret",
    }


@pytest.mark.asyncio
async def test_exchange_raises_on_error(monkeypatch):
    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        kc.httpx,
        "AsyncClient",
        lambda **kw: real_client(
            transport=httpx.MockTransport(lambda r: httpx.Response(400, json={"error": "invalid_token"})), **kw
        ),
    )
    with pytest.raises(httpx.HTTPStatusError):
        await kc.Keycloak(Settings(), scope=SCOPE).exchange("bad", "insurance-billing")


def test_caller_to_context():
    caller = kc.Caller("jane", "Jane Smith", ["B67890"], ["insurance-admin"])
    assert caller.to_context("tok") == {
        "user_id": "jane",
        "name": "Jane Smith",
        "policy_numbers": ["B67890"],
        "roles": ["insurance-admin"],
        "access_token": "tok",
    }


def test_assert_policy_access():
    assert kc.assert_policy_access("A12345") is None
    token = kc.caller_var.set(kc.Caller("jane", "Jane Smith", ["B67890"], []))
    try:
        assert kc.assert_policy_access(" b67890 ") is None
        assert kc.assert_policy_access("A12345") == (
            "Not authorised: policy A12345 does not belong to the authenticated customer."
        )
    finally:
        kc.caller_var.reset(token)


def test_identity_line():
    assert kc.identity_line(None) == ""
    assert kc.identity_line({"user_id": "demo-user-1", "name": ""}) == ""
    assert kc.identity_line({"name": "John Doe", "policy_numbers": ["A12345", "D1"]}) == (
        "Authenticated customer: John Doe, policies: A12345, D1\n"
    )


def _token_without(rsa_key, monkeypatch, *drop):
    class SigningKey:
        def __init__(self, key):
            self.key = key

    monkeypatch.setattr(
        kc.PyJWKClient,
        "get_signing_key_from_jwt",
        lambda self, token: SigningKey(rsa_key.public_key()),
    )
    claims = {
        "preferred_username": "john",
        "name": "John Doe",
        "aud": "insurance-triage",
        "scp": SCOPE,
        "policy_numbers": ["A12345"],
        "roles": [],
        "exp": int(time.time()) + 300,
    }
    for key in drop:
        del claims[key]
    return jwt.encode(claims, rsa_key, algorithm="RS256", headers={"kid": "test"})


def test_validate_rejects_token_without_exp(rsa_key, monkeypatch):
    token = _token_without(rsa_key, monkeypatch, "exp")
    with pytest.raises(jwt.InvalidTokenError):
        kc.Keycloak(Settings(), scope=SCOPE).validate(token)


def test_validate_rejects_token_without_preferred_username(rsa_key, monkeypatch):
    token = _token_without(rsa_key, monkeypatch, "preferred_username")
    with pytest.raises(jwt.InvalidTokenError):
        kc.Keycloak(Settings(), scope=SCOPE).validate(token)
