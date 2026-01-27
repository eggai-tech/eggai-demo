from eggai import Agent, Channel

from libraries.communication.channels import channels
from libraries.communication.messaging import AgentName, MessageType, subscribe
from libraries.communication.streaming import publish_error_message
from libraries.observability.logger import get_console_logger
from libraries.observability.tracing import TracedMessage, create_tracer, traced_handler
from libraries.observability.tracing.init_metrics import init_token_metrics

from .config import settings
from .types import ChatMessage
from .utils import get_conversation_string, process_billing_request

AGENT_NAME = AgentName.BILLING
billing_agent = Agent(name=AGENT_NAME)
logger = get_console_logger("billing_agent.handler")
agents_channel = Channel(channels.agents)
human_channel = Channel(channels.human)
human_stream_channel = Channel(channels.human_stream)
tracer = create_tracer("billing_agent")

init_token_metrics(
    port=settings.prometheus_metrics_port, application_name=settings.app_name
)

@subscribe(
    agent=billing_agent,
    channel=agents_channel,
    message_type=MessageType.BILLING_REQUEST,
    group_id="billing_agent_group",
)
@traced_handler("handle_billing_request")
async def handle_billing_request(msg: TracedMessage) -> None:
    """Handle incoming billing request messages from the agents channel."""
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
        logger.info(f"Processing billing request for connection {connection_id}")

        await process_billing_request(
            conversation_string,
            connection_id,
            str(msg.id),
            human_stream_channel,
            timeout_seconds=30.0,
        )

    except Exception as e:
        logger.error(f"Error in {AGENT_NAME}: {e}", exc_info=True)
        await publish_error_message(
            human_channel, AGENT_NAME,
            connection_id=locals().get("connection_id", "unknown"),
            traceparent=msg.traceparent if "msg" in locals() else None,
            tracestate=msg.tracestate if "msg" in locals() else None,
        )


@billing_agent.subscribe(channel=agents_channel)
async def handle_other_messages(msg: TracedMessage) -> None:
    """Handle non-billing messages received on the agent channel."""
    logger.debug("Received non-billing message: %s", msg)

if __name__ == "__main__":
    import asyncio

    from dspy import Prediction
    from dspy.streaming import StreamResponse

    from libraries.ml.dspy.language_model import dspy_set_language_model

    from .dspy_modules.billing import process_billing

    async def run():
        dspy_set_language_model(settings)

        from libraries.communication.channels import clear_channels
        await clear_channels()

        test_conversation = (
            "User: How much is my premium?\n"
            "BillingAgent: Could you please provide your policy number?\n"
            "User: It's B67890.\n"
        )

        logger.info("Running test query for billing agent")
        chunks = process_billing(chat_history=test_conversation)
        async for chunk in chunks:
            if isinstance(chunk, StreamResponse):
                logger.info(f"Chunk: {chunk.chunk}")
            elif isinstance(chunk, Prediction):
                logger.info(f"Final response: {chunk.final_response}")

    asyncio.run(run())
