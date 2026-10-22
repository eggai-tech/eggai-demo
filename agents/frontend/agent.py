import asyncio
import os
import uuid

import httpx
import jwt
import uvicorn
from eggai import Agent, Channel
from fastapi import FastAPI, Query
from opentelemetry import trace
from opentelemetry.context import Context
from starlette.websockets import WebSocket, WebSocketDisconnect

from libraries.communication.channels import channels
from libraries.communication.messaging import subscribe
from libraries.communication.protocol import AgentName, MessageType
from libraries.observability.logger import get_console_logger
from libraries.observability.tracing import TracedMessage, create_tracer
from libraries.observability.tracing.init_metrics import init_token_metrics
from libraries.observability.tracing.otel import (
    extract_span_context,
    safe_set_attribute,
    traced_handler,
)
from libraries.security.policy import notices

from .config import front_door, keycloak, settings
from .websocket_manager import WebSocketManager

logger = get_console_logger("frontend_agent")

GUARDRAILS_ENABLED = os.getenv("GUARDRAILS_TOKEN") is not None

if GUARDRAILS_ENABLED:
    try:
        from .guardrails import toxic_language_guard
    except ImportError as e:
        logger.error(f"Failed to import guardrails: {e}")
        toxic_language_guard = None
else:
    logger.info("Guardrails disabled (no GUARDRAILS_TOKEN)")
    toxic_language_guard = None


AGENT_NAME = AgentName.FRONTEND
frontend_agent = Agent(AGENT_NAME)
human_channel = Channel(channels.human)
human_stream_channel = Channel(channels.human_stream)
websocket_manager = WebSocketManager()

INTERACTION_NOTICE = "ai_interaction_notice"
NOTICES = {
    INTERACTION_NOTICE: (
        "You are chatting with an AI assistant, not a person. "
        "Its answers are generated automatically and can be wrong."
    ),
}
AI_LABEL = "AI-generated content"


def _refusal(decision: dict) -> str:
    return f"Refused by policy: {', '.join(decision['reasons'])}."


async def _disclose(connection_id: str, decision: dict) -> None:
    if INTERACTION_NOTICE in notices(decision):
        await websocket_manager.send_message_to_connection(
            connection_id,
            {"type": "notice", "notice": INTERACTION_NOTICE, "content": NOTICES[INTERACTION_NOTICE]},
        )
        websocket_manager.disclosures[connection_id].append(INTERACTION_NOTICE)


async def _checked_reply(connection_id: str, message_id: str, traceparent: str | None, text: str) -> dict:
    """The reply as the browser receives it: refused, or carrying the marks the decision asks for."""
    if not front_door.enabled:
        return {"content": text}
    decision = await front_door.decide(
        "output",
        websocket_manager.user_tokens.get(connection_id, ""),
        connection_id,
        message_id,
        traceparent or "",
        text,
        websocket_manager.disclosures.get(connection_id, []),
    )
    if not decision["allowed"]:
        return {"content": _refusal(decision), "refused": True}
    await _disclose(connection_id, decision)
    reply: dict[str, str | bool] = {"content": text}
    if "ai_generated_content_label" in notices(decision):
        reply["label"] = AI_LABEL
    if "synthetic_output_marking" in notices(decision):
        reply["ai_generated"] = True
    return reply


async def security_context(connection_id: str, token: str | None) -> dict:
    if not keycloak.enabled:
        return {
            "user_id": f"demo-user-{connection_id[:8]}",
            "name": "",
            "policy_numbers": [],
            "roles": [],
            "access_token": "",
        }
    caller = keycloak.validate(token or "")
    return caller.to_context(await keycloak.exchange(token or "", "insurance-triage"))


tracer = create_tracer("frontend_agent")
init_token_metrics(
    port=settings.prometheus_metrics_port, application_name=settings.app_name
)

def _extract_trace_context(connection_id: str) -> tuple[str, str, Context]:
    with tracer.start_as_current_span("frontend_chat", context=None) as root_span:
        root_ctx = root_span.get_span_context()
        traceparent = (
            f"00-{root_ctx.trace_id:032x}-{root_ctx.span_id:016x}-{root_ctx.trace_flags:02x}"
        )
        safe_set_attribute(root_span, "connection.id", str(connection_id))
        tracestate = str(root_ctx.trace_state) if root_ctx.trace_state else ""
    ctx = extract_span_context(traceparent, tracestate)
    # traceparent is built from a live SpanContext above, so it always parses.
    assert ctx is not None
    parent_ctx = trace.set_span_in_context(trace.NonRecordingSpan(ctx))
    return traceparent, tracestate, parent_ctx


async def _initialize_connection(websocket: WebSocket, connection_id: str) -> None:
    await websocket_manager.connect(websocket, connection_id)
    await websocket_manager.send_message_to_connection(
        connection_id, {"connection_id": connection_id}
    )


async def _process_user_messages(
    server: uvicorn.Server,
    websocket: WebSocket,
    connection_id: str,
    traceparent: str,
    tracestate: str,
) -> None:
    while True:
        try:
            data = await asyncio.wait_for(websocket.receive_json(), timeout=1)
        except TimeoutError:
            if server.should_exit:
                await websocket_manager.disconnect(connection_id)
                for conn in server.server_state.connections or []:
                    if hasattr(conn, "shutdown"):
                        conn.shutdown()
                break
            continue

        message_id = str(uuid.uuid4())
        content = data.get("payload")

        try:
            context = await security_context(connection_id, data.get("token"))
        except (jwt.PyJWTError, httpx.HTTPError) as e:
            logger.warning(f"Authentication failed for {connection_id}: {e}")
            await websocket_manager.send_message_to_connection(
                connection_id,
                {
                    "sender": "System",
                    "content": f"Authentication failed: {e}",
                    "type": MessageType.ASSISTANT_MESSAGE.value,
                },
            )
            continue

        token = data.get("token") or ""
        websocket_manager.user_tokens[connection_id] = token
        if front_door.enabled:
            decision = await front_door.decide(
                "inbound",
                token,
                connection_id,
                message_id,
                traceparent,
                content,
                websocket_manager.disclosures[connection_id],
            )
            if not decision["allowed"]:
                await websocket_manager.send_message_to_connection(
                    connection_id,
                    {
                        "sender": "Policy",
                        "content": _refusal(decision),
                        "refused": True,
                        "type": MessageType.ASSISTANT_MESSAGE.value,
                    },
                )
                continue
            await _disclose(connection_id, decision)

        if GUARDRAILS_ENABLED and toxic_language_guard:
            valid = await toxic_language_guard(content)
            if valid is None:
                await human_channel.publish(
                    TracedMessage(
                        id=uuid.UUID(message_id),
                        source=AGENT_NAME,
                        type=MessageType.AGENT_MESSAGE.value,
                        data={
                            "message": "Sorry, I can't help you with that.",
                            "connection_id": connection_id,
                            "agent": AgentName.TRIAGE,
                        },
                        traceparent=traceparent,
                        tracestate=tracestate,
                    )
                )
                continue
        else:
            valid = content

        await websocket_manager.attach_message_id(message_id, connection_id)
        websocket_manager.chat_messages[connection_id].append(
            {"role": "user", "content": valid, "id": message_id, "agent": "User"}
        )
        await human_channel.publish(
            TracedMessage(
                id=uuid.UUID(message_id),
                source=AGENT_NAME,
                type=MessageType.USER_MESSAGE.value,
                data={
                    "chat_messages": websocket_manager.chat_messages[connection_id],
                    "connection_id": connection_id,
                    "security_context": context,
                },
                traceparent=traceparent,
                tracestate=tracestate,
            )
        )


@tracer.start_as_current_span("add_websocket_gateway")
def add_websocket_gateway(route: str, app: FastAPI, server: uvicorn.Server) -> None:
    @app.websocket(route)
    async def websocket_handler(
        websocket: WebSocket, connection_id: str = Query(None, alias="connection_id")
    ):
        if server.should_exit:
            websocket.state.closed = True
            return

        if connection_id is None:
            connection_id = str(uuid.uuid4())

        traceparent, tracestate, parent_ctx = _extract_trace_context(connection_id)

        with tracer.start_as_current_span(
            "websocket_connection", context=parent_ctx, kind=trace.SpanKind.SERVER
        ) as span:
            try:
                safe_set_attribute(span, "connection.id", str(connection_id))
                await _initialize_connection(websocket, connection_id)
                await _process_user_messages(
                    server, websocket, connection_id, traceparent, tracestate
                )
            except WebSocketDisconnect:
                logger.info(f"WebSocket disconnected: {connection_id}")
            except Exception as e:
                logger.error(
                    f"Error with WebSocket {connection_id}: {e}", exc_info=True
                )
            finally:
                await websocket_manager.disconnect(connection_id)
                logger.info(f"WebSocket connection {connection_id} closed.")


@subscribe(agent=frontend_agent, channel=human_stream_channel)
async def handle_human_stream_messages(message: TracedMessage):
    message_type = message.type
    agent = message.source
    message_id = message.data.get("message_id")
    connection_id: str = message.data.get("connection_id", "unknown")

    if message_type == MessageType.AGENT_MESSAGE_STREAM_START.value:
        logger.info(f"Starting stream for message {message_id} from {agent}")
        await websocket_manager.send_message_to_connection(
            connection_id,
            {
                "sender": agent,
                "content": "",
                "message_id": message_id,
                "type": MessageType.ASSISTANT_MESSAGE_STREAM_START.value,
            },
        )

    elif message_type == MessageType.AGENT_MESSAGE_STREAM_WAITING_MESSAGE.value:
        waiting_text = message.data.get("message")
        await websocket_manager.send_message_to_connection(
            connection_id,
            {
                "sender": agent,
                "content": waiting_text,
                "message_id": message_id,
                "type": MessageType.ASSISTANT_MESSAGE_STREAM_WAITING_MESSAGE.value,
            },
        )

    elif message_type == MessageType.AGENT_MESSAGE_STREAM_CHUNK.value:
        # Under the front door a reply reaches the browser only whole, after its output check.
        if front_door.enabled:
            return
        chunk = message.data.get("message_chunk", "")
        chunk_index = message.data.get("chunk_index")
        await websocket_manager.send_message_to_connection(
            connection_id,
            {
                "sender": agent,
                "content": chunk,
                "chunk_index": chunk_index,
                "type": MessageType.ASSISTANT_MESSAGE_STREAM_CHUNK.value,
            },
        )
    elif message_type == MessageType.AGENT_MESSAGE_STREAM_END.value:
        reply = await _checked_reply(
            connection_id, str(message_id), message.traceparent, message.data.get("message", "")
        )
        websocket_manager.chat_messages[connection_id].append(
            {
                "role": "assistant",
                "content": reply["content"],
                "agent": agent,
                "id": str(message_id),
            }
        )
        await websocket_manager.send_message_to_connection(
            connection_id,
            {"sender": agent, **reply, "type": MessageType.ASSISTANT_MESSAGE_STREAM_END.value},
        )


@subscribe(agent=frontend_agent, channel=human_channel)
@traced_handler("handle_human_messages")
async def handle_human_messages(message: TracedMessage):
    message_type = message.type
    agent = message.data.get("agent")
    connection_id: str = message.data.get("connection_id", "unknown")
    message_id = message.id

    if message_type == MessageType.AGENT_MESSAGE.value:
        reply = await _checked_reply(
            connection_id, str(message_id), message.traceparent, message.data.get("message", "")
        )
        websocket_manager.chat_messages[connection_id].append(
            {
                "role": "assistant",
                "content": reply["content"],
                "agent": agent,
                "id": message_id,
            }
        )

        await websocket_manager.send_message_to_connection(
            connection_id,
            {"sender": agent, **reply, "type": MessageType.ASSISTANT_MESSAGE.value},
        )
