import asyncio
import json
import os

import httpx
import jwt
import pytest
import websockets

from libraries.security.keycloak import Keycloak

pytestmark = pytest.mark.integration

KEYCLOAK = os.environ.get("KEYCLOAK_URL", "http://localhost:8180")
CHAT_WS_URL = os.environ.get("CHAT_WS_URL", "ws://localhost:8000/ws")
TOKEN_URL = f"{KEYCLOAK}/realms/insurance/protocol/openid-connect/token"


def login(username: str) -> str:
    response = httpx.post(
        TOKEN_URL,
        data={"grant_type": "password", "client_id": "insurance-web", "username": username, "password": "insurance"},
    )
    response.raise_for_status()
    return response.json()["access_token"]


async def ask(token: str, text: str) -> str:
    async with websockets.connect(CHAT_WS_URL) as ws:
        await ws.send(json.dumps({"payload": text, "token": token}))
        while True:
            msg = json.loads(await asyncio.wait_for(ws.recv(), 120))
            if msg.get("type") in ("assistant_message_stream_end", "assistant_message"):
                return msg["content"]


@pytest.mark.asyncio
async def test_john_cannot_read_janes_policy():
    reply = await ask(login("john"), "How much is my premium? My policy number is B67890.")
    assert "B67890" in reply and ("not" in reply.lower() or "cannot" in reply.lower())


@pytest.mark.asyncio
async def test_john_reads_his_own_policy_without_giving_the_number():
    reply = await ask(login("john"), "How much is my premium?")
    assert "500" in reply


@pytest.mark.asyncio
async def test_triage_token_is_rejected_by_billing():
    class Frontend:
        keycloak_url = KEYCLOAK
        keycloak_realm = "insurance"
        keycloak_client_id = "insurance-frontend"
        keycloak_client_secret = "dev-frontend-secret"

    class Billing(Frontend):
        keycloak_client_id = "insurance-billing"
        keycloak_client_secret = ""

    triage_token = await Keycloak(Frontend(), scope="").exchange(login("john"), "insurance-triage")
    with pytest.raises(jwt.InvalidTokenError):
        Keycloak(Billing(), scope="api://insurance-billing/Billing.ReadWrite").validate(triage_token)
