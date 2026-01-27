from collections.abc import Callable
from typing import Any

from eggai import Agent, Channel
from opentelemetry import trace

from agents.triage.classifiers import get_classifier
from agents.triage.config import GROUP_ID, settings
from agents.triage.dspy_modules.small_talk import chatty
from agents.triage.models import AGENT_REGISTRY, TargetAgent
from libraries.communication.channels import channels
from libraries.communication.messaging import AgentName, MessageType, OffsetReset, subscribe
from libraries.communication.streaming import get_conversation_string, stream_dspy_response
from libraries.observability.logger import get_console_logger
from libraries.observability.tracing import (
    TracedMessage,
    create_tracer,
    format_span_as_traceparent,
    traced_handler,
)
from libraries.observability.tracing.otel import safe_set_attribute

AGENT_NAME = AgentName.TRIAGE
triage_agent = Agent(name=AGENT_NAME)
human_channel = Channel(channels.human)
human_stream_channel = Channel(channels.human_stream)
agents_channel = Channel(channels.agents)

tracer = create_tracer("triage_agent")
logger = get_console_logger("triage_agent.handler")


def get_current_classifier() -> Callable[..., Any]:
    """Get the classifier based on the configured version using the unified registry."""
    return get_classifier(settings.classifier_version)


current_classifier = get_current_classifier()


def build_conversation_string(chat_messages: list[dict[str, str]]) -> str:
    """Build a conversation history string from a list of chat entries.

    Delegates to the shared ``get_conversation_string`` helper using
    the ``"agent"`` key so that conversation lines show the real agent
    name (e.g. ``"Billing: ..."`` instead of ``"assistant: ..."``).
    """
    return get_conversation_string(chat_messages, role_key="agent")


async def _publish_to_agent(
    conversation_string: str, target_agent: TargetAgent, msg: TracedMessage
) -> None:
    """Publish a single classified user message to a specialized agent."""
    logger.info(f"Routing message to {target_agent}")
    with tracer.start_as_current_span("publish_to_agent") as span:
        safe_set_attribute(span, "target_agent", str(target_agent))
        safe_set_attribute(span, "conversation_length", len(conversation_string))
        child_traceparent, child_tracestate = format_span_as_traceparent(span)
        triage_to_agent_messages = [
            {
                "role": "user",
                "content": f"{conversation_string} \n{target_agent}: ",
            }
        ]
        await agents_channel.publish(
            TracedMessage(
                type=AGENT_REGISTRY[target_agent]["message_type"],
                source=AGENT_NAME,
                data={
                    "chat_messages": triage_to_agent_messages,
                    "message_id": msg.id,
                    "connection_id": msg.data.get("connection_id", "unknown"),
                },
                traceparent=child_traceparent,
                tracestate=child_tracestate,
            )
        )


async def _stream_chatty_response(
    conversation_string: str, msg: TracedMessage
) -> None:
    """Stream a 'chatty' (small-talk) response back to the human stream channel."""
    chunks = chatty(chat_history=conversation_string)
    await stream_dspy_response(
        chunks=chunks,
        agent_name=AGENT_NAME,
        connection_id=msg.data.get("connection_id", "unknown"),
        message_id=str(msg.id),
        stream_channel=human_stream_channel,
        tracer=tracer,
        span_name="chatty_stream_response",
        response_field="response",
    )


@subscribe(
    agent=triage_agent,
    channel=human_channel,
    message_type=MessageType.USER_MESSAGE,
    group_id=GROUP_ID,
    auto_offset_reset=OffsetReset.LATEST,
)
@traced_handler("handle_user_message")
async def handle_user_message(msg: TracedMessage) -> None:
    """Main handler for user messages: classify and route or stream chatty responses."""
    chat_messages: list[dict[str, Any]] = msg.data.get("chat_messages", [])
    connection_id: str = msg.data.get("connection_id", "unknown")

    span = trace.get_current_span()
    safe_set_attribute(span, "connection_id", connection_id)
    safe_set_attribute(span, "num_chat_messages", len(chat_messages))

    logger.info(f"Received message from connection {connection_id}")
    if not chat_messages:
        safe_set_attribute(span, "empty_chat_history", True)
        logger.warning("Empty chat history for connection %s", connection_id)
        await human_channel.publish(
            TracedMessage(
                type=MessageType.AGENT_MESSAGE,
                source=AGENT_NAME,
                data={
                    "message": "Sorry, I didn't receive any message to process.",
                    "connection_id": connection_id,
                    "agent": AGENT_NAME,
                },
            )
        )
        return

    conversation_string = build_conversation_string(chat_messages)
    safe_set_attribute(span, "conversation_length", len(conversation_string))

    try:
        response = current_classifier.classify(chat_history=conversation_string)
        safe_set_attribute(span, "classifier_version", settings.classifier_version)
        safe_set_attribute(span, "target_agent", str(response.target_agent))
        safe_set_attribute(span, "classification_latency_ms", response.latency_ms)
    except ValueError as e:
        safe_set_attribute(span, "error", str(e))
        logger.error("Classifier configuration error: %s", e)
        await human_channel.publish(
            TracedMessage(
                type=MessageType.AGENT_MESSAGE,
                source=AGENT_NAME,
                data={
                    "message": f"Configuration error: {e}",
                    "connection_id": connection_id,
                    "agent": AGENT_NAME,
                },
            )
        )
        return
    except Exception as e:
        safe_set_attribute(span, "error", "classification_failure")
        logger.error("Error during classification: %s", e, exc_info=True)
        await human_channel.publish(
            TracedMessage(
                type=MessageType.AGENT_MESSAGE,
                source=AGENT_NAME,
                data={
                    "message": "Sorry, I encountered an error while classifying your message.",
                    "connection_id": connection_id,
                    "agent": AGENT_NAME,
                },
            )
        )
        return

    target_agent = response.target_agent
    logger.info(
        f"Classification completed in {response.latency_ms:.2f} ms, "
        f"target agent: {target_agent}, classifier: {settings.classifier_version}"
    )

    if target_agent != TargetAgent.ChattyAgent:
        try:
            await _publish_to_agent(conversation_string, target_agent, msg)
        except Exception as e:
            logger.error("Error routing to agent %s: %s", target_agent, e, exc_info=True)
            await human_channel.publish(
                TracedMessage(
                    type=MessageType.AGENT_MESSAGE,
                    source=AGENT_NAME,
                    data={
                        "message": "Sorry, I couldn't route your request due to an internal error.",
                        "connection_id": connection_id,
                        "agent": AGENT_NAME,
                    },
                )
            )
    else:
        try:
            await _stream_chatty_response(conversation_string, msg)
        except Exception as e:
            logger.error("Error streaming chatty response: %s", e, exc_info=True)
            await human_stream_channel.publish(
                TracedMessage(
                    type=MessageType.AGENT_MESSAGE_STREAM_END,
                    source=AGENT_NAME,
                    data={
                        "message_id": str(msg.id),
                        "connection_id": connection_id,
                        "agent": AGENT_NAME,
                        "message": "Sorry, I encountered an error generating a response.",
                    },
                )
            )


@triage_agent.subscribe(channel=human_channel)
async def handle_other_messages(msg: TracedMessage) -> None:
    """Fallback handler for other message types on the human channel."""
    logger.debug("Received message: %s", msg)
