# Observability and Monitoring

Full distributed tracing and metrics collection using OpenTelemetry with GenAI
semantic conventions.

## Overview

The system provides end-to-end observability across all agents:

- **Distributed Tracing**: Request flow across agents via OpenTelemetry
- **Metrics Collection**: Token usage, latency, costs via Prometheus
- **Log Aggregation**: Structured logging with trace correlation
- **GenAI Conventions**: Standard metrics for LLM operations

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Agents    │────▶│   OTel      │────▶│   Tempo     │
│             │     │  Collector  │     │  (Traces)   │
└─────────────┘     └──────┬──────┘     └─────────────┘
                          │
                          ▼
                   ┌─────────────┐     ┌─────────────┐
                   │ Prometheus  │────▶│  Grafana    │
                   │  (Metrics)  │     │ (Dashboards)│
                   └─────────────┘     └─────────────┘
```

## Service URLs

| Service    | URL                   | Purpose                         |
| ---------- | --------------------- | ------------------------------- |
| Grafana    | http://localhost:3000 | Dashboards, trace visualization |
| Prometheus | http://localhost:9090 | Metrics queries                 |
| Tempo      | (via Grafana)         | Trace storage                   |

## OpenTelemetry Integration

### Initialization

```python
from libraries.observability.tracing import init_telemetry, create_tracer

# Initialize at startup
init_telemetry(
    app_name="billing_agent",
    endpoint="http://localhost:4317"
)

# Create tracer for spans
tracer = create_tracer("billing_agent")
```

### Tracing Decorators

```python
from libraries.observability.tracing import traced_handler

@traced_handler("handle_billing_request")
async def handle_billing_request(msg):
    # Automatically creates span with timing
    result = await process(msg)
    return result
```

### Trace Context Propagation

All messages include W3C trace context headers:

```python
{
    "traceparent": "00-trace_id-span_id-01",
    "tracestate": "vendor=value"
}
```

Context is automatically extracted and propagated across agents.

## GenAI Semantic Conventions

Standard metrics for LLM operations following the
[OpenTelemetry Semantic Conventions for GenAI](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
specification.

### Token Usage Metrics

| Metric                             | Description                   |
| ---------------------------------- | ----------------------------- |
| `gen_ai.client.token.usage`        | Total tokens (input + output) |
| `gen_ai.client.token.usage.input`  | Input/prompt tokens           |
| `gen_ai.client.token.usage.output` | Output/completion tokens      |

### Operation Metrics

| Metric                              | Description       |
| ----------------------------------- | ----------------- |
| `gen_ai.client.operation.duration`  | LLM call latency  |
| `gen_ai.client.time_to_first_token` | Streaming latency |
| `gen_ai.client.request.count`       | Total requests    |
| `gen_ai.client.error.count`         | Failed requests   |

### Attributes

Each metric includes attributes:

```python
{
    "gen_ai.system": "openai",
    "gen_ai.request.model": "gpt-4o-mini",
    "gen_ai.operation.name": "chat",
    "agent.name": "BillingAgent"
}
```

## DSPy Tracing

Automatic tracing for DSPy operations:

```python
from libraries.observability.tracing import TracedReAct

# Wrapped DSPy module with tracing
model = TracedReAct(
    signature=BillingSignature,
    tools=[get_billing_info],
    name="billing_react",
    tracer=tracer
)
```

Traces include:

- ReAct iterations
- Tool calls and results
- LLM prompts and responses
- Token counts per step

## Cost Tracking

Automatic cost calculation based on model pricing:

```python
from libraries.observability.tracing.pricing import calculate_cost

# Pricing per 1M tokens (configurable)
PRICING = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 5.00, "output": 15.00},
}

cost = calculate_cost(
    model="gpt-4o-mini",
    input_tokens=1000,
    output_tokens=500
)
```

## Prometheus Metrics

### Agent Metrics

Each agent exposes metrics on its Prometheus port:

| Agent      | Metrics Port |
| ---------- | ------------ |
| Frontend   | 9091         |
| Triage     | 9092         |
| Billing    | 9093         |
| Claims     | 9094         |
| Policies   | 9095         |
| Escalation | 9096         |
| Audit      | 9097         |

### Key Metrics

```promql
# Request rate by agent
rate(agent_requests_total[5m])

# Average response time
histogram_quantile(0.95, rate(agent_response_duration_seconds_bucket[5m]))

# Token usage by model
sum(gen_ai_client_token_usage_total) by (model)

# Error rate
rate(agent_errors_total[5m]) / rate(agent_requests_total[5m])
```

## Grafana Dashboards

### Pre-built Dashboards

1. **Agent Overview**
   - Request throughput by agent
   - Response latency percentiles
   - Error rates

2. **LLM Usage**
   - Token consumption over time
   - Cost tracking by model
   - Time-to-first-token

3. **Infrastructure**
   - Kafka consumer lag
   - Vespa search latency
   - Temporal workflow status

### Trace Exploration

1. Open Grafana → Explore
2. Select Tempo data source
3. Search by:
   - Trace ID
   - Service name
   - Operation name
   - Duration

## Structured Logging

All logs include trace context for correlation:

```python
from libraries.observability.logger import get_console_logger

logger = get_console_logger("billing_agent")

# Logs automatically include trace_id
logger.info("Processing billing request", extra={
    "connection_id": connection_id,
    "query_type": "premium"
})
```

Log format:

```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "level": "INFO",
  "message": "Processing billing request",
  "trace_id": "abc123",
  "span_id": "def456",
  "service": "billing_agent",
  "connection_id": "user-123",
  "query_type": "premium"
}
```

## Configuration

### Environment Variables

```bash
# OpenTelemetry
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
OTEL_SERVICE_NAME=billing_agent

# Metrics
PROMETHEUS_METRICS_PORT=9093

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
```

### Telemetry Settings

```python
# libraries/observability/tracing/config.py
class TracingConfig:
    enabled: bool = True
    endpoint: str = "http://localhost:4317"
    sample_rate: float = 1.0  # 100% sampling
    capture_content: bool = False  # Privacy setting
```

## Key Files

```
libraries/observability/
├── logger/
│   ├── config.py
│   └── logger.py
└── tracing/
    ├── config.py
    ├── otel.py           # OpenTelemetry setup
    ├── dspy.py           # DSPy integration
    ├── pricing.py        # Cost calculation
    ├── schemas.py        # Metric definitions
    └── init_metrics.py   # Metric initialization
```

---

**Previous:** [Classifier Strategies](classifier-strategies.md) | **Next:**
[Streaming Architecture](streaming-architecture.md) | [Back to Index](README.md)
