from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass

import httpx
import jwt
from jwt import PyJWKClient

TOKEN_EXCHANGE_GRANT = "urn:ietf:params:oauth:grant-type:token-exchange"
ACCESS_TOKEN_TYPE = "urn:ietf:params:oauth:token-type:access_token"


@dataclass
class Caller:
    user_id: str
    name: str
    policy_numbers: list[str]
    roles: list[str]

    @classmethod
    def from_claims(cls, claims: dict) -> Caller:
        return cls(
            user_id=claims.get("preferred_username", claims["sub"]),
            name=claims.get("name", ""),
            policy_numbers=list(claims.get("policy_numbers", [])),
            roles=list(claims.get("roles", [])),
        )

    def to_context(self, access_token: str) -> dict:
        return {
            "user_id": self.user_id,
            "name": self.name,
            "policy_numbers": self.policy_numbers,
            "roles": self.roles,
            "access_token": access_token,
        }


caller_var: ContextVar[Caller | None] = ContextVar("caller", default=None)

_jwks_clients: dict[str, PyJWKClient] = {}


class Keycloak:
    def __init__(self, settings, scope: str):
        self.url = settings.keycloak_url
        self.realm = settings.keycloak_realm
        self.client_id = settings.keycloak_client_id
        self.client_secret = settings.keycloak_client_secret
        self.scope = scope

    @property
    def enabled(self) -> bool:
        return bool(self.url)

    @property
    def token_endpoint(self) -> str:
        return f"{self.url}/realms/{self.realm}/protocol/openid-connect/token"

    def _jwks(self) -> PyJWKClient:
        uri = f"{self.url}/realms/{self.realm}/protocol/openid-connect/certs"
        if uri not in _jwks_clients:
            _jwks_clients[uri] = PyJWKClient(uri)
        return _jwks_clients[uri]

    def validate(self, token: str) -> Caller:
        key = self._jwks().get_signing_key_from_jwt(token)
        claims = jwt.decode(token, key.key, algorithms=["RS256"], audience=self.client_id, leeway=5)
        if self.scope not in claims.get("scp", "").split():
            raise jwt.InvalidTokenError(f"token missing scope {self.scope}")
        return Caller.from_claims(claims)

    async def exchange(self, token: str, audience: str) -> str:
        data = {
            "grant_type": TOKEN_EXCHANGE_GRANT,
            "subject_token": token,
            "subject_token_type": ACCESS_TOKEN_TYPE,
            "requested_token_type": ACCESS_TOKEN_TYPE,
            "audience": audience,
            "scope": f"{audience}-scope",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
        }
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(self.token_endpoint, data=data)
        response.raise_for_status()
        return response.json()["access_token"]


def assert_policy_access(policy_number: str) -> str | None:
    caller = caller_var.get()
    number = policy_number.strip().upper()
    if caller is None or number in caller.policy_numbers:
        return None
    return f"Not authorised: policy {number} does not belong to the authenticated customer."


def identity_line(security_context: dict | None) -> str:
    if not security_context or not security_context.get("name"):
        return ""
    policies = ", ".join(security_context.get("policy_numbers", []))
    return f"Authenticated customer: {security_context['name']}, policies: {policies}\n"
