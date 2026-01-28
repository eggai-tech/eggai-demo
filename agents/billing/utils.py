from eggai import Channel

from libraries.communication.channels import channels
from libraries.communication.messaging import AgentName
from libraries.communication.streaming import (
    get_conversation_string as get_conversation_string,
)
from libraries.communication.streaming import (
    stream_dspy_response,
    validate_conversation,
)
from libraries.observability.logger import get_console_logger
from libraries.observability.tracing import create_tracer

from .dspy_modules.billing import process_billing
from .types import ModelConfig

default_human_stream_channel = Channel(channels.human_stream)

logger = get_console_logger("billing_agent.utils")
tracer = create_tracer("billing_agent")


async def process_billing_request(
    conversation_string: str,
    connection_id: str,
    message_id: str,
    human_stream_channel: Channel = default_human_stream_channel,
    timeout_seconds: float | None = None,
) -> None:
    config = ModelConfig(name="billing_react", timeout_seconds=timeout_seconds or 30.0)
    with tracer.start_as_current_span("process_billing_request") as span:
        validate_conversation(conversation_string, tracer, span)

        chunks = process_billing(chat_history=conversation_string, config=config)
        await stream_dspy_response(
            chunks=chunks,
            agent_name=AgentName.BILLING,
            connection_id=connection_id,
            message_id=message_id,
            stream_channel=human_stream_channel,
            tracer=tracer,
        )
