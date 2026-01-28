# Streaming Architecture

Real-time token-by-token response streaming from agents to users via WebSocket
and Kafka.

## Overview

The streaming architecture enables:

- **Low Perceived Latency**: Users see responses as they generate
- **Durable Delivery**: All chunks flow through Kafka for reliability
- **Progress Feedback**: "Thinking..." indicators before generation starts

## Architecture

```
┌──────────┐    ┌──────────┐    ┌─────────────┐    ┌──────────┐
│   LLM    │───▶│  Agent   │───▶│ human_stream│───▶│ Frontend │
│ (tokens) │    │(streaming)│    │  (Kafka)    │    │(WebSocket)│
└──────────┘    └──────────┘    └─────────────┘    └──────────┘
     │                                                    │
     │              Token-by-token flow                   │
     └────────────────────────────────────────────────────┘
```

## Message Flow

### 1. Waiting Indicator

Before LLM generation starts:

```json
{
  "type": "assistant_message_stream_waiting_message",
  "sender": "PoliciesAgent",
  "message_id": "msg-123",
  "content": "Looking up your policy information..."
}
```

### 2. Stream Start

When generation begins:

```json
{
  "type": "assistant_message_stream_start",
  "sender": "PoliciesAgent",
  "message_id": "msg-123"
}
```

### 3. Stream Chunks

Token-by-token delivery:

```json
{
  "type": "assistant_message_stream_chunk",
  "sender": "PoliciesAgent",
  "message_id": "msg-123",
  "content": "Your policy",
  "chunk_index": 0
}
```

```json
{
  "type": "assistant_message_stream_chunk",
  "content": " covers fire damage",
  "chunk_index": 1
}
```

### 4. Stream End

Final message with complete content:

```json
{
  "type": "assistant_message_stream_end",
  "sender": "PoliciesAgent",
  "message_id": "msg-123",
  "content": "Your policy covers fire damage up to $500,000..."
}
```

## Implementation

### Agent Side (Streaming Response)

```python
from libraries.communication.streaming import stream_dspy_response

# DSPy streaming
chunks = dspy.streamify(
    model,
    stream_listeners=[
        dspy.streaming.StreamListener(signature_field_name="final_response"),
    ],
    async_streaming=True,
)(chat_history=conversation)

# Send to Kafka human_stream channel
await stream_dspy_response(
    chunks=chunks,
    agent_name=AGENT_NAME,
    connection_id=connection_id,
    message_id=message_id,
    stream_channel=human_stream_channel,
    tracer=tracer,
)
```

### Frontend Side (WebSocket Handler)

```javascript
socket.onmessage = (event) => {
  const data = JSON.parse(event.data);

  switch (data.type) {
    case "assistant_message_stream_waiting_message":
      showThinkingIndicator(data.content);
      break;

    case "assistant_message_stream_start":
      initializeMessageBubble(data.message_id);
      break;

    case "assistant_message_stream_chunk":
      appendToMessage(data.message_id, data.content);
      break;

    case "assistant_message_stream_end":
      finalizeMessage(data.message_id, data.content);
      break;
  }
};
```

## Kafka Integration

### Channel Configuration

The `human_stream` channel is configured for real-time delivery:

```python
from eggai import Channel
from libraries.communication.channels import channels

human_stream_channel = Channel(channels.human_stream)
```

### Message Publishing

```python
async def publish_chunk(channel, chunk_data):
    await channel.publish(
        message={
            "type": "assistant_message_stream_chunk",
            "sender": agent_name,
            "message_id": message_id,
            "content": chunk,
            "chunk_index": index,
        },
        headers=trace_headers
    )
```

### Consumer Groups

Frontend subscribes with consumer group for reliability:

```python
@subscribe(
    agent=frontend_agent,
    channel=human_stream_channel,
    group_id="frontend_stream_group"
)
async def handle_stream(msg):
    await websocket.send(msg)
```

## WebSocket Gateway

### Connection Management

```python
class WebSocketManager:
    def __init__(self):
        self.connections: dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket) -> str:
        connection_id = str(uuid.uuid4())
        self.connections[connection_id] = websocket
        return connection_id

    async def send_to_connection(self, connection_id: str, message: dict):
        if ws := self.connections.get(connection_id):
            await ws.send_json(message)
```

### Stream Routing

Messages are routed to the correct WebSocket by `connection_id`:

```python
@subscribe(channel=human_stream_channel)
async def handle_stream_message(msg):
    connection_id = msg.data.get("connection_id")
    await websocket_manager.send_to_connection(
        connection_id,
        msg.data
    )
```

## UI Rendering

### Thinking Indicator

```javascript
// Bouncing dots animation
<div className="flex gap-1">
  <span className="animate-bounce" style={{ animationDelay: "0ms" }}>.</span>
  <span className="animate-bounce" style={{ animationDelay: "150ms" }}>.</span>
  <span className="animate-bounce" style={{ animationDelay: "300ms" }}>.</span>
</div>;
```

### Streaming Text

```javascript
// Cursor animation during streaming
<span>{streamedText}</span>
<span className="animate-pulse">|</span>
```

### Message Finalization

```javascript
// Replace streaming state with final content
setMessages((prev) => [
  ...prev,
  { sender: agent, text: finalContent, id: messageId },
]);
```

## Error Handling

### Stream Interruption

If streaming is interrupted:

1. Agent detects disconnection
2. Partial content is preserved in Kafka
3. Frontend can request resync on reconnect

### Timeout Handling

```python
# Configure stream timeout
STREAM_TIMEOUT = 120  # seconds

async def stream_with_timeout(chunks):
    async with asyncio.timeout(STREAM_TIMEOUT):
        async for chunk in chunks:
            yield chunk
```

### Reconnection

```javascript
socket.onclose = () => {
  setTimeout(() => {
    initWebSocket(); // Auto-reconnect
  }, 3000);
};
```

## Performance Considerations

### Chunk Size

- Small chunks (1-5 tokens): Smoother animation, more messages
- Large chunks (10-20 tokens): Fewer messages, chunkier display
- **Recommendation**: Let LLM determine natural boundaries

### Backpressure

```python
# Handle slow consumers
async def publish_with_backpressure(channel, message):
    try:
        await asyncio.wait_for(
            channel.publish(message),
            timeout=5.0
        )
    except asyncio.TimeoutError:
        logger.warning("Consumer slow, buffering...")
```

### Metrics

Track streaming performance:

| Metric                    | Description                |
| ------------------------- | -------------------------- |
| `stream_chunks_total`     | Total chunks sent          |
| `stream_duration_seconds` | Time from start to end     |
| `time_to_first_chunk`     | Latency before first token |

## Key Files

```
libraries/communication/
├── streaming.py          # stream_dspy_response()
└── protocol/messages.py  # StreamChunkData, StreamEndData

agents/frontend/
├── websocket_manager.py  # WebSocket connection handling
└── public/index.html     # React streaming UI
```

---

**Previous:** [Observability](observability.md) | **Next:**
[Model Flexibility](model-flexibility.md) | [Back to Index](README.md)
