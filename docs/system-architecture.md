# Multi-Agent System Architecture

## Overview

A distributed insurance support system where specialized AI agents collaborate
via event-driven messaging to handle user inquiries.

## Architecture Diagram

```mermaid
graph TB
    subgraph "User Layer"
        UI[Web Chat :8000]
    end
    
    subgraph "Routing Layer"
        WS[Frontend Agent]
        TR[Triage Agent]
    end
    
    subgraph "Specialized Agents"
        BI[Billing Agent]
        CL[Claims Agent]
        PO["Policies Agent (RAG)"]
        ES[Escalation Agent]
    end

    subgraph "Infrastructure"
        RP["Message Broker (Redpanda)"]
        VE["Vector Search (Vespa)"]
        TE["Workflows (Temporal)"]
    end
    
    UI <-->|WebSocket| WS
    WS -->|human channel| TR
    TR -->|agents channel| BI & CL & PO & ES
    
    WS & TR & BI & CL & PO & ES -.->|pub/sub| RP
    PO <-->|search| VE
    PO <-->|ingestion| TE

    style UI fill:#e1f5fe
    style WS fill:#f3e5f5
    style TR fill:#f3e5f5
    style BI fill:#f3e5f5
    style CL fill:#f3e5f5
    style PO fill:#f3e5f5
    style ES fill:#f3e5f5
    style RP fill:#fff3e0
    style VE fill:#fff3e0
    style TE fill:#fff3e0
```

## Core Components

### Message Channels

| Channel        | Purpose         | Publishers | Subscribers        |
| -------------- | --------------- | ---------- | ------------------ |
| `human`        | User messages   | Frontend   | Triage, Audit      |
| `human_stream` | Agent responses | All agents | Frontend           |
| `agents`       | Routed requests | Triage     | Specialized agents |
| `audit_logs`   | Compliance      | Audit      | Monitoring systems |

### Agent Roles

**Triage Agent**

- Classifies user intent using ML models
- Routes to appropriate specialist
- Handles greetings directly

**Specialized Agents**

- **Billing**: Premium amounts, payment dates, billing updates
- **Claims**: File claims, check status, update information
- **Policies**: Document search via RAG, coverage questions
- **Escalation**: Complaints, complex issues, ticket creation

**Audit Agent**

- Monitors all channels
- Creates compliance logs

## Message Flow

```
1. User sends message via WebSocket
2. Triage classifies intent
3. Routes to specialized agent
4. Agent processes and responds
5. Response streams back to user
6. Audit logs all interactions
```

## Key Design Patterns

### Event-Driven Communication

- Asynchronous message passing via Redpanda
- No direct agent-to-agent calls
- Enables independent scaling

### Stateless Agents

- No shared state between instances
- All context in messages
- Horizontal scaling ready

### RAG Integration

- Policies agent uses Vespa for semantic search
- Temporal orchestrates document ingestion
- Hybrid search (keyword + vector)

## Interoperability

In enterprise environments, diverse programming languages and frameworks create
fragmentation. EggAI Agents serve as thin, flexible connectors, enabling seamless
integration within the multi-agent system without costly re-platforming.

### Framework Agnostic

Agents can use any AI framework internally:

- **DSPy**: Programmatic prompting and optimization
- **LangChain**: Chains and retrieval pipelines
- **LiteLLM**: Unified LLM API interface
- **LlamaIndex**: Document indexing and retrieval

### Language Interoperability

Using Kafka transport, agents can be implemented in any language with a Kafka client:

| Language | Library |
|----------|---------|
| Python | `confluent-kafka` |
| JavaScript/TypeScript | `kafkajs` |
| Java/Kotlin | `kafka-clients` |
| Go | `sarama` |
| C# | `Confluent.Kafka` |
| Rust | `rdkafka` |

### Standardized Message Format

For structured communication, the system uses the CloudEvents specification
(see [Multi-Agent Communication](multi-agent-communication.md)), ensuring
consistency and interoperability across all agents regardless of implementation
language or framework.

## Platform Services

| Service    | Purpose                         | Port  |
| ---------- | ------------------------------- | ----- |
| Redpanda   | Kafka-compatible message broker | 19092 |
| Vespa      | Vector search for documents     | 8080  |
| Temporal   | Durable workflow orchestration  | 7233  |
| PostgreSQL | Temporal/MLflow storage         | 5432  |
| Grafana    | Metrics visualization           | 3000  |
| Prometheus | Metrics collection              | 9090  |

## Deployment Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Agent 1   │     │   Agent 2   │     │   Agent N   │
│  Container  │     │  Container  │     │  Container  │
└──────┬──────┘     └──────┬──────┘     └──────┬──────┘
       │                   │                   │
       └───────────────────┴───────────────────┘
                           │
                    ┌──────▼──────┐
                    │  Redpanda   │
                    │   Cluster   │
                    └─────────────┘
```

Each agent:

- Runs as independent container
- Scales horizontally
- Connects to shared infrastructure

---

**Next:** [Agents Overview](agents-overview.md)
