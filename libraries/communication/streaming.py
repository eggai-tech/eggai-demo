"""
Shared streaming utilities for agent response delivery.

Provides common patterns for streaming DSPy responses to the frontend
via Kafka channels. All domain agents (billing, claims, escalation,
policies) use the same streaming protocol:

  1. Publish stream_start
  2. Publish stream_chunk for each DSPy StreamResponse
  3. Publish stream_end with the final Prediction

Usage:
    from libraries.communication.streaming import stream_dspy_response

    await stream_dspy_response(
        chunks=process_billing(chat_history=conversation),
        agent_name="Billing",
        connection_id=connection_id,
        message_id=message_id,
        stream_channel=human_stream_channel,
        tracer=tracer,
    )
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from dspy import Prediction
from dspy.streaming import StreamResponse
from eggai import Channel

from libraries.communication.protocol import MessageType
from libraries.observability.logger import get_console_logger
from libraries.observability.tracing import (
    TracedMessage,
    format_span_as_traceparent,
)
from libraries.observability.tracing.otel import safe_set_attribute

logger = get_console_logger("streaming")


def get_conversation_string(
    chat_messages: list[dict[str, Any]],
    tracer: Any | None = None,
    *,
    role_key: str = "role",
) -> str:
    """
    Format chat messages into a conversation string for DSPy input.

    Args:
        chat_messages: List of chat message dicts with 'role' and 'content' keys.
        tracer: Optional OpenTelemetry tracer for span creation.
        role_key: Dict key to use for the speaker identifier.
            Defaults to ``"role"`` (e.g. ``"user"``/``"assistant"``).
            The triage agent passes ``"agent"`` to get the actual agent name.

    Returns:
        Formatted conversation string like "User: hello\\nAssistant: hi\\n".
    """
    if tracer is not None:
        with tracer.start_as_current_span("get_conversation_string") as span:
            return _build_conversation_string(chat_messages, span, role_key=role_key)
    return _build_conversation_string(chat_messages, role_key=role_key)


def _build_conversation_string(
    chat_messages: list[dict[str, Any]],
    span: Any | None = None,
    *,
    role_key: str = "role",
) -> str:
    if span is not None:
        safe_set_attribute(span, "chat_messages_count", len(chat_messages) if chat_messages else 0)

    if not chat_messages:
        if span is not None:
            safe_set_attribute(span, "empty_messages", True)
        return ""

    conversation_parts: list[str] = []
    for chat in chat_messages:
        if "content" not in chat:
            if span is not None:
                safe_set_attribute(span, "invalid_message", True)
            logger.warning("Message missing content field")
            continue

        content = chat["content"]
        if content is not None and not content:
            continue

        role = chat.get(role_key, "User")
        conversation_parts.append(f"{role}: {content}")

    conversation = "\n".join(conversation_parts) + "\n"
    if span is not None:
        safe_set_attribute(span, "conversation_length", len(conversation))
    return conversation


async def stream_dspy_response(
    chunks: AsyncIterator[Any],
    agent_name: str,
    connection_id: str,
    message_id: str,
    stream_channel: Channel,
    tracer: Any,
    *,
    span_name: str | None = None,
    response_field: str = "final_response",
) -> None:
    """
    Stream a DSPy async generator response to the frontend.

    Publishes stream_start, stream_chunk, and stream_end messages
    following the standard EggAI streaming protocol.

    Args:
        chunks: Async iterator from a DSPy streaming call.
        agent_name: Name of the agent (e.g. "Billing", "Claims").
        connection_id: WebSocket connection ID.
        message_id: Unique message ID.
        stream_channel: Channel to publish streaming messages to.
        tracer: OpenTelemetry tracer for the calling agent.
        span_name: Optional custom span name (defaults to "stream_response").
        response_field: Prediction attribute containing the final response.
            Defaults to ``"final_response"``.  The triage chatty module
            uses ``"response"`` instead.
    """
    _span_name = span_name or "stream_response"

    with tracer.start_as_current_span(_span_name) as span:
        child_traceparent, child_tracestate = format_span_as_traceparent(span)
        safe_set_attribute(span, "connection_id", connection_id)
        safe_set_attribute(span, "message_id", message_id)

        # 1. Stream start
        await stream_channel.publish(
            TracedMessage(
                type=MessageType.AGENT_MESSAGE_STREAM_START,
                source=agent_name,
                data={
                    "message_id": message_id,
                    "connection_id": connection_id,
                },
                traceparent=child_traceparent,
                tracestate=child_tracestate,
            )
        )
        logger.info(f"Stream started for message {message_id}")

        chunk_count = 0
        try:
            async for chunk in chunks:
                if isinstance(chunk, StreamResponse):
                    chunk_count += 1
                    await stream_channel.publish(
                        TracedMessage(
                            type=MessageType.AGENT_MESSAGE_STREAM_CHUNK,
                            source=agent_name,
                            data={
                                "message_chunk": chunk.chunk,
                                "message_id": message_id,
                                "chunk_index": chunk_count,
                                "connection_id": connection_id,
                            },
                            traceparent=child_traceparent,
                            tracestate=child_tracestate,
                        )
                    )
                elif isinstance(chunk, Prediction):
                    response = getattr(chunk, response_field, None) or ""
                    response = response.replace(" [[ ## completed ## ]]", "")

                    logger.info(
                        f"Sending stream end with response: "
                        f"{response[:100] if response else 'EMPTY'}"
                    )
                    await stream_channel.publish(
                        TracedMessage(
                            type=MessageType.AGENT_MESSAGE_STREAM_END,
                            source=agent_name,
                            data={
                                "message_id": message_id,
                                "message": response,
                                "agent": agent_name,
                                "connection_id": connection_id,
                            },
                            traceparent=child_traceparent,
                            tracestate=child_tracestate,
                        )
                    )
                    logger.info(f"Stream ended for message {message_id}")

            safe_set_attribute(span, "chunk_count", chunk_count)

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Error in streaming response: {e}", exc_info=True)
            safe_set_attribute(span, "error", str(e))
            span.set_status(1, str(e))
            await stream_channel.publish(
                TracedMessage(
                    type=MessageType.AGENT_MESSAGE_STREAM_END,
                    source=agent_name,
                    data={
                        "message_id": message_id,
                        "message": "I'm sorry, I encountered an error while processing your request.",
                        "agent": agent_name,
                        "connection_id": connection_id,
                    },
                    traceparent=child_traceparent,
                    tracestate=child_tracestate,
                )
            )


async def publish_error_message(
    channel: Channel,
    agent_name: str,
    connection_id: str,
    message: str = "I apologize, but I'm having trouble processing your request right now. Please try again.",
    traceparent: str | None = None,
    tracestate: str | None = None,
) -> None:
    """
    Publish a standardized error message to a channel.

    Used by domain agents to send user-facing error messages when
    chat history is empty or an exception occurs during processing.
    """
    await channel.publish(
        TracedMessage(
            type=MessageType.AGENT_MESSAGE,
            source=agent_name,
            data={
                "message": message,
                "connection_id": connection_id,
                "agent": agent_name,
            },
            traceparent=traceparent,
            tracestate=tracestate,
        )
    )


async def publish_waiting_message(
    channel: Channel,
    agent_name: str,
    connection_id: str,
    message_id: str,
    message: str = "Thinking...",
) -> None:
    """
    Publish a waiting message to give the user immediate feedback.

    Sends an ``AGENT_MESSAGE_STREAM_WAITING_MESSAGE`` so the frontend can
    display a "thinking" or "connecting" indicator while the agent processes.
    """
    await channel.publish(
        TracedMessage(
            type=MessageType.AGENT_MESSAGE_STREAM_WAITING_MESSAGE,
            source=agent_name,
            data={
                "message_id": message_id,
                "connection_id": connection_id,
                "message": message,
            },
        )
    )


def validate_conversation(conversation_string: str, tracer: Any, span: Any) -> None:
    """
    Validate that a conversation string is non-empty and long enough.

    Args:
        conversation_string: The conversation to validate.
        tracer: OpenTelemetry tracer.
        span: Current span for attribute recording.

    Raises:
        ValueError: If conversation is empty or too short.
    """
    safe_set_attribute(span, "conversation_length", len(conversation_string))
    if not conversation_string or len(conversation_string.strip()) < 5:
        safe_set_attribute(span, "error", "Empty or too short conversation")
        span.set_status(1, "Invalid input")
        raise ValueError("Conversation history is too short to process")
