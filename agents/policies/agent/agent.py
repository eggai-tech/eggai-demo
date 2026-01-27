from eggai import Agent, Channel

from agents.policies.agent.config import (
    AGENT_NAME,
    CONSUMER_GROUP_ID,
    model_config,
    settings,
)
from agents.policies.agent.reasoning import process_policies
from agents.policies.agent.types import ChatMessage
from libraries.communication.channels import channels
from libraries.communication.messaging import MessageType, subscribe
from libraries.communication.streaming import (
    get_conversation_string,
    publish_error_message,
    stream_dspy_response,
    validate_conversation,
)
from libraries.observability.logger import get_console_logger
from libraries.observability.tracing import TracedMessage, create_tracer, traced_handler
from libraries.observability.tracing.init_metrics import init_token_metrics

policies_agent = Agent(name=AGENT_NAME)
logger = get_console_logger("policies_agent.handler")
agents_channel = Channel(channels.agents)
human_channel = Channel(channels.human)
human_stream_channel = Channel(channels.human_stream)
tracer = create_tracer("policies_agent")

init_token_metrics(
    port=settings.prometheus_metrics_port, application_name=settings.app_name
)


async def process_policy_request(
    conversation_string: str,
    connection_id: str,
    message_id: str,
    timeout_seconds: float | None = None,
) -> None:
    """Generate a response to a policy request with streaming output."""
    config = model_config
    if timeout_seconds:
        config = model_config.model_copy(update={"timeout_seconds": timeout_seconds})

    with tracer.start_as_current_span("process_policy_request") as span:
        validate_conversation(conversation_string, tracer, span)

        chunks = process_policies(chat_history=conversation_string, config=config)
        await stream_dspy_response(
            chunks=chunks,
            agent_name=AGENT_NAME,
            connection_id=connection_id,
            message_id=message_id,
            stream_channel=human_stream_channel,
            tracer=tracer,
        )


@subscribe(
    agent=policies_agent,
    channel=agents_channel,
    message_type=MessageType.POLICY_REQUEST,
    group_id=CONSUMER_GROUP_ID,
)
@traced_handler("handle_policy_request")
async def handle_policy_request(msg: TracedMessage) -> None:
    """Handle incoming policy request messages from the agents channel."""
    try:
        chat_messages: list[ChatMessage] = msg.data.get("chat_messages", [])
        connection_id: str = msg.data.get("connection_id", "unknown")

        if not chat_messages:
            logger.warning(f"Empty chat history for connection: {connection_id}")
            await publish_error_message(
                human_channel, AGENT_NAME, connection_id,
                message="I apologize, but I didn't receive any message content to process.",
                traceparent=msg.traceparent, tracestate=msg.tracestate,
            )
            return

        conversation_string = get_conversation_string(chat_messages, tracer=tracer)
        logger.info(f"Processing policy request for connection {connection_id}")

        await process_policy_request(
            conversation_string, connection_id, str(msg.id), timeout_seconds=30.0
        )

    except Exception as e:
        logger.error(f"Error in {AGENT_NAME}: {e}", exc_info=True)
        await publish_error_message(
            human_channel, AGENT_NAME,
            connection_id=locals().get("connection_id", "unknown"),
            traceparent=msg.traceparent if "msg" in locals() else None,
            tracestate=msg.tracestate if "msg" in locals() else None,
        )


@policies_agent.subscribe(channel=agents_channel)
async def handle_other_messages(msg: TracedMessage) -> None:
    """Handle non-policy messages received on the agent channel."""
    logger.debug("Received non-policy message: %s", msg)


if __name__ == "__main__":
    import asyncio

    from libraries.ml.dspy.language_model import dspy_set_language_model

    async def run():
        dspy_set_language_model(settings)

        from libraries.communication.channels import clear_channels
        await clear_channels()

        test_conversation = (
            "User: I need information about my policy.\n"
            "PoliciesAgent: Sure, I can help with that. Could you please provide me with your policy number?\n"
            "User: My policy number is A12345\n"
        )

        logger.info("Running test query for policies agent")
        await process_policy_request(
            test_conversation, "test-connection", "test-message", timeout_seconds=30.0
        )

    asyncio.run(run())
