# Building Agents Guide

Build a new agent using DSPy ReAct in 5 minutes. This tutorial creates a
hypothetical GDPR agent to demonstrate the pattern used by all agents in the
system.

## Step 1: Create the Agent Structure

```bash
mkdir -p agents/gdpr
```

## Step 2: Create Configuration File

Create `agents/gdpr/config.py`:

```python
from pydantic import Field
from pydantic_settings import SettingsConfigDict

from libraries.communication.messaging import AgentName
from libraries.core import BaseAgentConfig

AGENT_NAME = "GDPRAgent"
CONSUMER_GROUP_ID = "gdpr_agent_group"


class Settings(BaseAgentConfig):
    app_name: str = Field(default="gdpr_agent")
    prometheus_metrics_port: int = Field(default=9098)

    model_config = SettingsConfigDict(
        env_prefix="GDPR_", env_file=".env", env_ignore_empty=True, extra="ignore"
    )


settings = Settings()
```

## Step 3: Create the Agent

Create `agents/gdpr/agent.py`:

```python
import dspy
from eggai import Agent, Channel

from libraries.communication.channels import channels
from libraries.communication.messaging import MessageType, subscribe
from libraries.communication.streaming import stream_dspy_response
from libraries.observability.logger import get_console_logger
from libraries.observability.tracing import (
    TracedMessage,
    TracedReAct,
    create_tracer,
    traced_handler,
)

from .config import AGENT_NAME, CONSUMER_GROUP_ID, settings

logger = get_console_logger("gdpr_agent")
gdpr_agent = Agent(name=AGENT_NAME)
agents_channel = Channel(channels.agents)
human_channel = Channel(channels.human)
human_stream_channel = Channel(channels.human_stream)
tracer = create_tracer("gdpr_agent")

# Sample GDPR data
GDPR_ARTICLES = {
    "17": {
        "title": "Right to erasure ('right to be forgotten')",
        "content": "The data subject shall have the right to obtain from the controller the erasure of personal data...",
    },
    "33": {
        "title": "Notification of a personal data breach",
        "content": "In case of a personal data breach, the controller shall notify within 72 hours...",
    },
}


# Define tools for ReAct
def list_gdpr_articles() -> str:
    """List all GDPR article numbers and titles."""
    lines = [
        f"Article {num}: {data['title']}"
        for num, data in sorted(GDPR_ARTICLES.items(), key=lambda x: int(x[0]))
    ]
    return "GDPR Articles:\n" + "\n".join(lines)


def read_gdpr_article(article_number: str) -> str:
    """Read a specific GDPR article content."""
    if article_number in GDPR_ARTICLES:
        article = GDPR_ARTICLES[article_number]
        return f"Article {article_number}: {article['title']}\n\n{article['content']}"
    return f"Article {article_number} not found"


# Create ReAct signature
class GDPRSignature(dspy.Signature):
    """Answer GDPR-related questions using available article tools."""

    chat_history: str = dspy.InputField(desc="Full conversation context.")
    final_response: str = dspy.OutputField(desc="Answer based on GDPR knowledge.")


# Create ReAct agent
gdpr_model = TracedReAct(
    GDPRSignature,
    tools=[list_gdpr_articles, read_gdpr_article],
    name="gdpr_react",
    tracer=tracer,
    max_iters=3,
)


@subscribe(
    agent=gdpr_agent,
    channel=agents_channel,
    message_type=MessageType.GDPR_REQUEST,  # Add to MessageType enum
    group_id=CONSUMER_GROUP_ID,
)
@traced_handler("handle_gdpr_request")
async def handle_gdpr_request(msg: TracedMessage) -> None:
    """Handle GDPR inquiries using ReAct."""
    chat_messages = msg.data.get("chat_messages", [])
    connection_id = msg.data.get("connection_id", "unknown")

    if not chat_messages:
        logger.warning(f"Empty chat history for connection: {connection_id}")
        return

    conversation_string = "\n".join(
        f"{m.get('role', 'User')}: {m['content']}" for m in chat_messages if "content" in m
    )

    chunks = dspy.streamify(
        gdpr_model,
        stream_listeners=[
            dspy.streaming.StreamListener(signature_field_name="final_response"),
        ],
        include_final_prediction_in_output_stream=True,
        is_async_program=False,
        async_streaming=True,
    )(chat_history=conversation_string)

    await stream_dspy_response(
        chunks=chunks,
        agent_name=AGENT_NAME,
        connection_id=connection_id,
        message_id=str(msg.id),
        stream_channel=human_stream_channel,
        tracer=tracer,
    )


@gdpr_agent.subscribe(channel=agents_channel)
async def handle_other_messages(msg: TracedMessage) -> None:
    """Handle non-GDPR messages received on the agent channel."""
    logger.debug("Received non-GDPR message: %s", msg)
```

## Step 4: Create Main Entry Point

Create `agents/gdpr/main.py`:

```python
import asyncio

from eggai import eggai_main
from eggai.transport import eggai_set_default_transport

from libraries.communication.transport import create_kafka_transport
from libraries.ml.dspy.language_model import dspy_set_language_model
from libraries.observability.logger import get_console_logger
from libraries.observability.tracing import init_telemetry

from .config import settings

eggai_set_default_transport(
    lambda: create_kafka_transport(
        bootstrap_servers=settings.kafka_bootstrap_servers,
        ssl_cert=settings.kafka_ca_content,
    )
)

from .agent import gdpr_agent  # noqa: E402 (must import after transport setup)

logger = get_console_logger("gdpr_agent")


@eggai_main
async def main():
    logger.info(f"Starting {settings.app_name}")
    init_telemetry(app_name=settings.app_name, endpoint=settings.otel_endpoint)
    dspy_set_language_model(settings)

    await gdpr_agent.start()
    logger.info(f"{settings.app_name} started successfully")

    await asyncio.Future()  # Keep running


if __name__ == "__main__":
    asyncio.run(main())
```

## Step 5: Create Module Init

Create `agents/gdpr/__init__.py`:

```python
# GDPR Agent Module
```

## Step 6: Run Your Agent

```bash
uv run -m agents.gdpr.main
```

## What You Built

A minimal GDPR agent following the same patterns as all production agents:

- **ReAct reasoning**: DSPy TracedReAct with OpenTelemetry tracing
- **Streaming responses**: Uses `stream_dspy_response()` for token-by-token
  output
- **Kafka messaging**: Subscribes via `@subscribe` with `MessageType` filtering
- **Ready to extend**: Add more GDPR articles or connect to real data sources

## Next Steps

To integrate with the main system:

1. Add `GDPR_REQUEST` to the `MessageType` enum in
   `libraries/communication/protocol/enums.py`
2. Add `GDPRAgent` to the triage agent's `AGENT_REGISTRY` in
   `agents/triage/models.py`
3. Add routing logic in the triage classifier

---

**Previous:** [Vespa Search Guide](vespa-search-guide.md) | **Next:**
[Retrieval Performance Testing](retrieval-performance-testing.md) |
[Back to Index](README.md)
