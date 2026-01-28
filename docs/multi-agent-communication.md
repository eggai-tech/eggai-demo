# Multi-Agent Communication

This document describes how agents communicate within the multi-agent insurance
support system using event-driven architecture and the EggAI SDK.

## Overview

The multi-agent system uses an event-driven architecture where agents
communicate asynchronously through channels. This design enables loose coupling,
scalability, and resilience.

## Communication Infrastructure

### Kafka/Redpanda Message Bus

The system uses Redpanda (Kafka-compatible) as the central message bus for all
agent communication:

- **Broker**: Kafka bootstrap servers
- **Protocol**: Event streaming with consumer groups
- **Message Format**: CloudEvents JSON with W3C Trace Context
- **Delivery Guarantees**: At-least-once delivery

### Message Protocol (CloudEvents)

All messages follow the [CloudEvents specification](https://cloudevents.io/) (v1.0), an
open standard for describing event data in a common way. This ensures interoperability
and makes it easy to integrate with other event-driven systems.

#### Message Structure

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "type": "com.eggai.billing.request",
  "source": "triage-agent",
  "specversion": "1.0",
  "datacontenttype": "application/json",
  "time": "2024-01-15T10:30:00Z",
  "subject": "connection-123",
  "traceparent": "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01",
  "tracestate": "eggai=session123",
  "data": {
    "connection_id": "conn-456",
    "chat_messages": [
      {"role": "user", "content": "What is my bill?", "agent": null}
    ],
    "security_context": {
      "user_id": "user-789",
      "tenant_id": "tenant-abc",
      "consent_scope": ["billing_read"],
      "retention_policy": "30d"
    }
  }
}
```

#### CloudEvents Fields

| Field | Required | Description |
|-------|----------|-------------|
| `id` | Yes | Unique event identifier (UUID) |
| `type` | Yes | Event type (e.g., `com.eggai.billing.request`) |
| `source` | Yes | Event producer (agent name) |
| `specversion` | Yes | CloudEvents spec version (`1.0`) |
| `datacontenttype` | No | Payload content type (`application/json`) |
| `time` | No | Event timestamp (ISO 8601) |
| `subject` | No | Event subject (connection ID) |
| `data` | Yes | Event payload |

#### W3C Trace Context Extensions

| Field | Description |
|-------|-------------|
| `traceparent` | W3C Trace Context propagation header |
| `tracestate` | Vendor-specific trace information |

These headers enable distributed tracing across all agents via OpenTelemetry.

### Communication Channels

The system defines four primary channels:

1. **`human`** - User messages from the frontend
   - Producers: Frontend Agent
   - Consumers: Triage Agent, Audit Agent

2. **`human_stream`** - Streaming responses to users
   - Producers: All specialized agents
   - Consumers: Frontend Agent, Audit Agent

3. **`agents`** - Inter-agent communication
   - Producers: Triage Agent, specialized agents
   - Consumers: All agents (filtered by message type)

4. **`audit_logs`** - Compliance and monitoring
   - Producers: Audit Agent
   - Consumers: External monitoring systems

## EggAI SDK Framework

### Channel-Based Communication

Agents use the EggAI SDK to subscribe to channels and handle messages:

```python
from eggai import Agent

agent = Agent(
    name="billing",
    transport_config={"bootstrap_servers": "localhost:19092"}
)

@agent.subscribe(channel="agents", type="BillingInquiry")
async def handle_billing_inquiry(message: BillingInquiry):
    # Process billing inquiry
    response = await process_billing(message)
    
    # Send response back
    await agent.publish(
        channel="human_stream",
        message=response,
        correlation_id=message.correlation_id
    )
```

### Message Types

Each agent defines strongly-typed messages using Pydantic:

```python
class BillingInquiry(BaseMessage):
    query: str
    policy_number: Optional[str]
    customer_id: str
```

### Message Filtering

Agents can filter messages by:

- **Type**: Only receive specific message types
- **Source**: Filter by sending agent
- **Headers**: Custom routing logic

## Message Flow Patterns

### 1. User Request Flow

```
User → Frontend Agent → [human] → Triage Agent
                                      ↓
                                 [agents] → Specialized Agent
                                              ↓
                                        [human_stream] → Frontend → User
```

### 2. Inter-Agent Collaboration

```
Claims Agent → [agents] → Policies Agent (get policy details)
                 ↓
         [agents] ← Response
```

### 3. Streaming Responses

Agents can stream partial results using the `human_stream` channel:

```python
async for chunk in generate_response():
    await agent.publish(
        channel="human_stream",
        message=StreamChunk(content=chunk),
        correlation_id=correlation_id
    )
```

## Distributed Tracing

All messages include distributed tracing headers:

- **Trace ID**: Unique identifier for the entire conversation
- **Span ID**: Identifier for individual operations
- **Parent Span**: Links related operations

This enables end-to-end observability through Grafana and Tempo.

## Best Practices

### 1. Message Design

- Keep messages small and focused
- Include correlation IDs for request tracking
- Use versioning for message schema evolution

### 2. Channel Usage

- Use `agents` channel for internal communication
- Reserve `human_stream` for user-facing responses
- Implement proper error messages for user communication

### 3. Error Handling

- Implement retry with exponential backoff (3 attempts)
- Consider dead letter queues for failed messages
- Use circuit breakers to prevent cascading failures
- Log errors for debugging and monitoring

### 4. Performance

- Batch messages when possible
- Use streaming for large responses
- Implement proper backpressure handling

### 5. Security

- Validate all incoming messages
- Sanitize user input
- Use SSL/TLS for production deployments

## Configuration

Message bus configuration via environment variables:

```bash
# Kafka/Redpanda Connection
KAFKA_BOOTSTRAP_SERVERS=localhost:19092

# Security (Production)
KAFKA_SECURITY_PROTOCOL=SSL
KAFKA_SSL_CA_LOCATION=/path/to/ca-cert
KAFKA_SSL_CERT_LOCATION=/path/to/cert
KAFKA_SSL_KEY_LOCATION=/path/to/key
```

## Monitoring

Monitor agent communication through:

- **Redpanda Console**: Message flow visualization
- **Grafana Dashboards**: Throughput and latency metrics
- **Prometheus Metrics**: Consumer lag and error rates
- **Distributed Tracing**: End-to-end request tracking

---

**Previous:** [Agents Overview](agents-overview.md) | **Next:**
[Classifier Strategies](classifier-strategies.md) | [Back to Index](README.md)
