from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from agents.frontend import agent as agent_mod
from agents.frontend import main as main_mod
from libraries.communication.protocol import MessageType
from libraries.observability.tracing import TracedMessage

NOTICE = {"type": "disclose", "notice": "ai_interaction_notice"}
ALLOW = {"allowed": True, "verdict": "allow", "controls": [], "reasons": [], "obligations": []}
DENY = {"allowed": False, "verdict": "deny", "controls": ["INP-001"], "reasons": ["input_invalid"], "obligations": []}
LABELLED = {
    **ALLOW,
    "verdict": "transform",
    "obligations": [
        {"type": "disclose", "notice": "ai_generated_content_label"},
        {"type": "disclose", "notice": "synthetic_output_marking"},
    ],
}


@pytest.fixture
def front_door(monkeypatch):
    monkeypatch.setattr(agent_mod.front_door, "url", "http://127.0.0.1:8181/v1/data/foundry/frontdoor/decision")
    monkeypatch.setattr(agent_mod.keycloak, "url", "")
    send = AsyncMock()
    monkeypatch.setattr(agent_mod.websocket_manager, "send_message_to_connection", send)
    publish = AsyncMock()
    monkeypatch.setattr(agent_mod.human_channel, "publish", publish)
    return send, publish


def _websocket(*messages):
    websocket = MagicMock()
    websocket.receive_json = AsyncMock(side_effect=[*messages, WebSocketDisconnect()])
    return websocket


async def _run(websocket, connection_id):
    server = MagicMock()
    server.should_exit = False
    with pytest.raises(WebSocketDisconnect):
        await agent_mod._process_user_messages(server, websocket, connection_id, "00-tp", "ts")


@pytest.mark.asyncio
async def test_inbound_sends_the_notice_once_then_publishes(front_door, monkeypatch):
    send, publish = front_door
    calls = []

    async def decide(point, token, connection_id, message_id, traceparent, text, shown):
        calls.append((point, token, text, list(shown)))
        return ALLOW if "ai_interaction_notice" in shown else {**ALLOW, "verdict": "transform", "obligations": [NOTICE]}

    monkeypatch.setattr(agent_mod.front_door, "decide", decide)

    await _run(_websocket({"payload": "hi", "token": "tok"}, {"payload": "again", "token": "tok"}), "conn-in")

    assert calls == [
        ("inbound", "tok", "hi", []),
        ("inbound", "tok", "again", ["ai_interaction_notice"]),
    ]
    notices = [c.args[1] for c in send.await_args_list if c.args[1].get("type") == "notice"]
    assert notices == [{
        "type": "notice", "notice": "ai_interaction_notice", "content": agent_mod.NOTICES["ai_interaction_notice"],
    }]
    assert publish.await_count == 2


@pytest.mark.asyncio
async def test_inbound_refusal_is_not_published(front_door, monkeypatch):
    send, publish = front_door
    monkeypatch.setattr(agent_mod.front_door, "decide", AsyncMock(return_value=DENY))

    await _run(_websocket({"payload": "", "token": "tok"}), "conn-deny")

    send.assert_awaited_once_with("conn-deny", {
        "sender": "Policy",
        "content": "Refused by policy: input_invalid.",
        "refused": True,
        "type": MessageType.ASSISTANT_MESSAGE.value,
    })
    publish.assert_not_awaited()


def _stream(message_type, **data):
    return TracedMessage(
        type=message_type,
        source="PoliciesAgent",
        data={"message_id": "msg-1", "connection_id": "conn-out", **data},
        traceparent="00-reply",
    )


@pytest.mark.asyncio
async def test_output_labels_the_whole_reply_and_holds_chunks(front_door, monkeypatch):
    send, _ = front_door
    decide = AsyncMock(return_value=LABELLED)
    monkeypatch.setattr(agent_mod.front_door, "decide", decide)
    agent_mod.websocket_manager.user_tokens["conn-out"] = "tok"

    await agent_mod.handle_human_stream_messages(
        _stream(MessageType.AGENT_MESSAGE_STREAM_CHUNK.value, message_chunk="Your ", chunk_index=1)
    )
    send.assert_not_awaited()

    await agent_mod.handle_human_stream_messages(
        _stream(MessageType.AGENT_MESSAGE_STREAM_END.value, message="Your policy is active.")
    )

    decide.assert_awaited_once_with("output", "tok", "conn-out", "msg-1", "00-reply", "Your policy is active.", [])
    send.assert_awaited_once_with("conn-out", {
        "sender": "PoliciesAgent",
        "content": "Your policy is active.",
        "label": agent_mod.AI_LABEL,
        "ai_generated": True,
        "type": MessageType.ASSISTANT_MESSAGE_STREAM_END.value,
    })
    agent_mod.websocket_manager.user_tokens.pop("conn-out")


@pytest.mark.asyncio
async def test_output_refusal_replaces_the_reply(front_door, monkeypatch):
    send, _ = front_door
    deny = {**DENY, "reasons": ["protected_content"]}
    monkeypatch.setattr(agent_mod.front_door, "decide", AsyncMock(return_value=deny))

    await agent_mod.handle_human_messages(TracedMessage(
        type=MessageType.AGENT_MESSAGE.value,
        source="BillingAgent",
        data={"message": "Policy B67890 is overdue.", "connection_id": "conn-ref", "agent": "BillingAgent"},
    ))

    sent = send.await_args.args[1]
    assert sent["content"] == "Refused by policy: protected_content."
    assert sent["refused"] is True
    assert agent_mod.websocket_manager.chat_messages["conn-ref"][-1]["content"] == sent["content"]
    agent_mod.websocket_manager.chat_messages.pop("conn-ref")


def test_admin_proxy_denied_by_policy(monkeypatch):
    monkeypatch.setattr(main_mod.front_door, "url", "http://127.0.0.1:8181/v1/data/foundry/frontdoor/decision")
    decide = AsyncMock(return_value={**DENY, "controls": ["IAM-004"], "reasons": ["mfa_required"]})
    monkeypatch.setattr(main_mod.front_door, "decide", decide)
    upstream = AsyncMock(return_value=httpx.Response(200, json={"ok": True}))
    monkeypatch.setattr(main_mod.upstream, "request", upstream)

    response = TestClient(main_mod.api).get("/api/claims/claims", headers={"Authorization": "Bearer t"})

    assert response.status_code == 403
    assert response.json()["detail"] == {"reasons": ["mfa_required"], "controls": ["IAM-004"]}
    assert decide.await_args.args == ("admin", "t")
    upstream.assert_not_awaited()
