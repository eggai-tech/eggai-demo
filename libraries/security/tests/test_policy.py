import json

import httpx
import pytest

from libraries.security import policy
from libraries.security.policy import UNAVAILABLE, FrontDoor, notices

URL = "http://opa/v1/data/foundry/frontdoor/decision"
ALLOW = {"allowed": True, "verdict": "allow", "controls": [], "reasons": [], "obligations": []}


@pytest.fixture
def opa(monkeypatch):
    seen = []
    replies = []
    client = httpx.AsyncClient

    def handler(request):
        seen.append(json.loads(request.content))
        return replies.pop(0)

    monkeypatch.setattr(
        policy.httpx, "AsyncClient", lambda **kw: client(transport=httpx.MockTransport(handler), **kw)
    )
    return seen, replies


@pytest.mark.asyncio
async def test_allow_sends_the_input_shape(opa):
    seen, replies = opa
    replies.append(httpx.Response(200, json={"result": ALLOW}))

    decision = await FrontDoor(URL).decide(
        "inbound", "browser-token", "conn-1", "msg-1", "00-abc-def-01", "hello", ["ai_interaction_notice"]
    )

    assert decision == ALLOW
    assert seen == [{"input": {
        "point": "inbound", "user_token": "browser-token", "connection_id": "conn-1", "message_id": "msg-1",
        "traceparent": "00-abc-def-01", "text": "hello", "disclosures_shown": ["ai_interaction_notice"],
    }}]


@pytest.mark.asyncio
async def test_deny_is_returned(opa):
    _, replies = opa
    deny = {**ALLOW, "allowed": False, "verdict": "deny", "reasons": ["mfa_required"]}
    replies.append(httpx.Response(200, json={"result": deny}))

    assert await FrontDoor(URL).decide("admin", "t") == deny


@pytest.mark.asyncio
async def test_disclose_obligations(opa):
    _, replies = opa
    obligations = [
        {"type": "disclose", "notice": "ai_generated_content_label"},
        {"type": "log"},
        {"type": "disclose", "notice": "synthetic_output_marking"},
    ]
    replies.append(httpx.Response(200, json={"result": {**ALLOW, "verdict": "transform", "obligations": obligations}}))

    decision = await FrontDoor(URL).decide("output", "t", text="reply")

    assert notices(decision) == ["ai_generated_content_label", "synthetic_output_marking"]


@pytest.mark.asyncio
async def test_no_result_is_deny(opa):
    _, replies = opa
    replies.append(httpx.Response(200, json={}))

    assert await FrontDoor(URL).decide("inbound", "t") == UNAVAILABLE


@pytest.mark.asyncio
async def test_error_status_is_deny(opa):
    _, replies = opa
    replies.append(httpx.Response(500, text="boom"))

    assert await FrontDoor(URL).decide("inbound", "t") == UNAVAILABLE


@pytest.mark.asyncio
async def test_unreachable_is_deny():
    decision = await FrontDoor("http://127.0.0.1:1/v1/data/foundry/frontdoor/decision").decide("inbound", "t")

    assert decision["allowed"] is False
    assert decision["reasons"] == ["policy_decision_unavailable"]


def test_enabled_by_url():
    assert FrontDoor(URL).enabled
    assert not FrontDoor("").enabled
