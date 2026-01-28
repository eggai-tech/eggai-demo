# Documentation

AI-powered insurance customer self-service built with multi-agent architecture.

**Use Case:** Policyholders get instant answers 24/7 without calling.

---

## Quick Start

```bash
python scripts/start.py    # Start all services
open http://localhost:8000  # Open chat UI
```

---

## Service URLs

| Service  | URL                              | Purpose            |
| -------- | -------------------------------- | ------------------ |
| Chat UI  | http://localhost:8000            | Main interface     |
| Admin    | http://localhost:8000/admin.html | System overview    |
| Grafana  | http://localhost:3000            | Metrics & traces   |
| Redpanda | http://localhost:8082            | Message streams    |
| Temporal | http://localhost:8081            | Workflow execution |
| Vespa    | http://localhost:19071           | Search index       |
| MLflow   | http://localhost:5001            | Model tracking     |

---

## Documentation Guide

### 1. Architecture

| Document                                        | Description                                   |
| ----------------------------------------------- | --------------------------------------------- |
| [System Architecture](./system-architecture.md) | High-level overview, components, message flow |
| [Agents Overview](./agents-overview.md)         | All 7 agents and their capabilities           |

### 2. Core Concepts

| Document                                                    | Description                                              |
| ----------------------------------------------------------- | -------------------------------------------------------- |
| [Multi-Agent Communication](./multi-agent-communication.md) | EggAI SDK, CloudEvents protocol, Redpanda messaging      |
| [Classifier Strategies](./classifier-strategies.md)         | 9 triage classifiers (v0-v8), from prompts to fine-tuned |
| [Observability](./observability.md)                         | OpenTelemetry, GenAI semantic conventions, Grafana       |
| [Streaming Architecture](./streaming-architecture.md)       | Real-time token streaming via WebSocket and Kafka        |
| [Model Flexibility](./model-flexibility.md)                 | LiteLLM multi-provider support (OpenAI, Azure, local)    |

### 3. RAG System

| Document                                      | Description                                     |
| --------------------------------------------- | ----------------------------------------------- |
| [Agentic RAG](./agentic-rag.md)               | ReAct pattern, tool selection, hybrid search    |
| [Ingestion Pipeline](./ingestion-pipeline.md) | Temporal workflows, document processing         |
| [Vespa Search Guide](./vespa-search-guide.md) | Hybrid search, ranking profiles, query examples |

### 4. Development

| Document                                                            | Description                          |
| ------------------------------------------------------------------- | ------------------------------------ |
| [Building Agents](./building-agents-eggai.md)                       | Create a new agent in 5 minutes      |
| [Retrieval Performance Testing](./retrieval-performance-testing.md) | RAG quality metrics and benchmarking |

### 5. Advanced Topics

| Document                                                                          | Description                            |
| --------------------------------------------------------------------------------- | -------------------------------------- |
| [Agent Optimization](./advanced-topics/agent-optimization.md)                     | DSPy COPRO/SIMBA, MLflow tracking      |
| [Multi-Environment Deployment](./advanced-topics/multi-environment-deployment.md) | Dev, staging, production configuration |

---

## Reading Order

**For new engineers**, read in this order:

1. [System Architecture](./system-architecture.md) - Understand the big picture
2. [Agents Overview](./agents-overview.md) - Know what each agent does
3. [Multi-Agent Communication](./multi-agent-communication.md) - Learn the
   messaging patterns
4. [Classifier Strategies](./classifier-strategies.md) - See the ML evolution
5. [Agentic RAG](./agentic-rag.md) - Understand the retrieval system
6. [Building Agents](./building-agents-eggai.md) - Create your own agent

---

_See the main [README.md](../README.md) for installation and setup
instructions._
