from eggai import Agent, Channel

from libraries.communication.channels import channels
from libraries.communication.messaging import MessageType, subscribe
from libraries.communication.streaming import (
    get_conversation_string,
    stream_dspy_response,
    validate_conversation,
)
from libraries.observability.logger import get_console_logger
from libraries.observability.tracing import TracedMessage, create_tracer, traced_handler
from libraries.observability.tracing.init_metrics import init_token_metrics

from .config import AGENT_NAME, GROUP_ID, settings
from .dspy_modules.escalation import process_escalation
from .types import ChatMessage

escalation_agent = Agent(name=AGENT_NAME)
logger = get_console_logger(AGENT_NAME)
agents_channel = Channel(channels.agents)
human_channel = Channel(channels.human)
human_stream_channel = Channel(channels.human_stream)
tracer = create_tracer(AGENT_NAME)

init_token_metrics(
    port=settings.prometheus_metrics_port, application_name=settings.app_name
)


async def process_escalation_request(
    conversation_string: str,
    connection_id: str,
    message_id: str,
    timeout_seconds: float | None = None,
) -> None:
    from .config import model_config

    config = model_config
    if timeout_seconds:
        config = model_config.model_copy(update={"timeout_seconds": timeout_seconds})

    with tracer.start_as_current_span("process_escalation_request") as span:
        validate_conversation(conversation_string, tracer, span)

        chunks = process_escalation(chat_history=conversation_string, config=config)
        await stream_dspy_response(
            chunks=chunks,
            agent_name=AGENT_NAME,
            connection_id=connection_id,
            message_id=message_id,
            stream_channel=human_stream_channel,
            tracer=tracer,
        )


@subscribe(
    agent=escalation_agent,
    channel=agents_channel,
    message_type=MessageType.ESCALATION_REQUEST,
    group_id=GROUP_ID,
)
@traced_handler("handle_ticketing_request")
async def handle_ticketing_request(msg: TracedMessage) -> None:
    try:
        chat_messages: list[ChatMessage] = msg.data.get("chat_messages", [])
        connection_id: str = msg.data.get("connection_id", "unknown")

        if not chat_messages:
            logger.warning(f"Empty chat history for connection: {connection_id}")
            await process_escalation_request(
                "User: [No message content received]",
                connection_id,
                str(msg.id),
                timeout_seconds=30.0,
            )
            return

        conversation_string = get_conversation_string(chat_messages, tracer=tracer)
        logger.info(f"Processing ticketing request for connection {connection_id}")

        await process_escalation_request(
            conversation_string, connection_id, str(msg.id), timeout_seconds=60.0
        )

    except Exception as e:
        logger.error(f"Error in {AGENT_NAME}: {e}", exc_info=True)
        try:
            connection_id = locals().get("connection_id", "unknown")
            message_id = str(msg.id) if msg else "unknown"
            await process_escalation_request(
                f"System: Error occurred - {str(e)}",
                connection_id,
                message_id,
                timeout_seconds=60.0,
            )
        except Exception:
            logger.error("Failed to send error response", exc_info=True)


@escalation_agent.subscribe(channel=agents_channel)
async def handle_other_messages(msg: TracedMessage) -> None:
    logger.debug("Received non-ticketing message: %s", msg)


if __name__ == "__main__":
    import asyncio

    from libraries.ml.dspy.language_model import dspy_set_language_model

    async def run():
        dspy_set_language_model(settings)

        from libraries.communication.channels import clear_channels
        await clear_channels()

        test_conversation = (
            "User: I need to escalate an issue with my policy A12345.\n"
            "TicketingAgent: I can help you with that. Let me check if there are any existing tickets for policy A12345.\n"
            "User: My claim was denied incorrectly and I need this reviewed by technical support. My email is john@example.com.\n"
        )

        logger.info("Running simplified escalation agent test")
        await process_escalation_request(
            test_conversation,
            "test-connection-123",
            "test-message-456",
            timeout_seconds=30.0,
        )

    asyncio.run(run())
