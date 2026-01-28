# Multi-Agent Insurance Support System

[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge&logo=opensourceinitiative&logoColor=white)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![EggAI SDK](https://img.shields.io/badge/Built%20with-EggAI-orange?style=for-the-badge)](https://github.com/eggai-tech/eggai)

A production-ready reference architecture for building enterprise AI agent
systems. Features 7 collaborating agents, 8 classifier strategies, RAG-powered
document search, and full observability.

![Chat UI Screenshot](https://raw.githubusercontent.com/eggai-tech/EggAI/refs/heads/main/docs/docs/assets/support-chat.png)

## Quick Start

```bash
# Clone and start (one command!)
git clone git@github.com:eggai-tech/eggai-demo.git
cd eggai-demo
make start
```

Open **http://localhost:8000** and start chatting.

> **Note:** Runs completely locally with LM Studio - no cloud services or API
> keys required!

## What This Demo Shows

| Feature                     | Description                                                    |
| --------------------------- | -------------------------------------------------------------- |
| **7 Collaborating Agents**  | Triage, Billing, Claims, Policies, Escalation, Audit, Frontend |
| **9 Classifier Strategies** | Compare LLM, fine-tuned, and neural network approaches (v0-v8) |
| **RAG Document Search**     | Vespa-powered hybrid search (70% semantic + 30% keyword)       |
| **Production Patterns**     | Health checks, observability, message-driven architecture      |
| **Full Observability**      | Grafana dashboards, distributed tracing, metrics               |

## Architecture at a Glance

```
User (WebSocket) → Frontend Agent → Triage Agent → Specialized Agent → Response
                                         │
                                    Classifier
                                    (v0-v8: pick your strategy)
```

**Message Flow:**

1. User sends message via WebSocket to **Frontend**
2. **Triage** classifies intent using configurable classifier (v0-v8)
3. Routes to specialized agent: **Billing**, **Claims**, **Policies**, or
   **Escalation**
4. **Audit** monitors all interactions for compliance
5. Response streams back to user

## Commands

| Command       | Description                                |
| ------------- | ------------------------------------------ |
| `make start`  | Start everything (infrastructure + agents) |
| `make stop`   | Stop agents                                |
| `make health` | Check service health                       |
| `make test`   | Run tests                                  |
| `make help`   | Show all commands                          |

## Explore the Codebase

### Entry Points

| Want to...            | Start here                                                     |
| --------------------- | -------------------------------------------------------------- |
| Understand the system | [docs/system-architecture.md](docs/system-architecture.md)     |
| See how agents work   | [agents/triage/agent.py](agents/triage/agent.py)               |
| Compare classifiers   | [agents/triage/classifiers/](agents/triage/classifiers/)       |
| Add a new agent       | [docs/building-agents-eggai.md](docs/building-agents-eggai.md) |
| Configure RAG search  | [docs/agentic-rag.md](docs/agentic-rag.md)                     |

### Code Organization

```
├── agents/                    # AI agents (self-contained modules)
│   ├── frontend/             # WebSocket gateway + chat UI
│   ├── triage/               # Classification and routing
│   │   ├── agent.py          # Message handler
│   │   ├── classifiers/      # 9 classifier strategies (v0-v8)
│   │   └── dspy_modules/     # DSPy-based classifiers
│   ├── billing/              # Payment inquiries
│   ├── claims/               # Claim processing
│   ├── policies/             # RAG-powered policy search
│   ├── escalation/           # Complex issue handling
│   └── audit/                # Compliance monitoring
├── libraries/                 # Shared utilities
│   ├── communication/        # Kafka channels
│   ├── observability/        # Logging, tracing, metrics
│   └── ml/                   # DSPy, MLflow integration
├── config/                    # Configuration
│   └── defaults.env          # Sensible defaults (works out of box)
├── scripts/                   # Startup utilities
│   ├── start.py              # One-command startup
│   ├── stop.py               # Graceful shutdown
│   └── health_check.py       # Service health checks
└── docs/                      # Documentation
```

## Classifier Comparison

The triage agent supports 9 classification strategies. Select via
`TRIAGE_CLASSIFIER_VERSION`:

| Version | Type              | Latency | API Call | Training |
| ------- | ----------------- | ------- | -------- | -------- |
| **v0**  | Minimal prompt    | ~500ms  | Yes      | No       |
| **v1**  | Enhanced prompt   | ~600ms  | Yes      | No       |
| **v2**  | COPRO optimized   | ~500ms  | Yes      | One-time |
| **v3**  | Few-shot MLflow   | ~50ms   | No       | Yes      |
| **v4**  | Zero-shot COPRO   | ~400ms  | Yes      | One-time |
| **v5**  | Attention network | ~20ms   | No       | Yes      |
| **v6**  | OpenAI fine-tuned | ~300ms  | Yes      | Yes      |
| **v7**  | Gemma fine-tuned  | ~100ms  | No       | Yes      |
| **v8**  | RoBERTa LoRA      | ~50ms   | No       | Yes      |

**Default:** v4 (configured in `config/defaults.env` — best balance of accuracy
and simplicity)

```python
# Using the unified classifier interface
from agents.triage.classifiers import get_classifier, list_classifiers

classifier = get_classifier("v4")
result = classifier.classify("User: What's my bill?")
print(result.target_agent)  # BillingAgent

# Compare all classifiers
for info in list_classifiers():
    print(f"{info.version}: {info.name}")
```

## Infrastructure Services

| Service          | URL                   | Description         |
| ---------------- | --------------------- | ------------------- |
| Chat UI          | http://localhost:8000 | Main application    |
| Redpanda Console | http://localhost:8082 | Message queue UI    |
| Vespa            | http://localhost:8080 | Vector search       |
| Temporal UI      | http://localhost:8081 | Workflow monitoring |
| MLflow           | http://localhost:5001 | Experiment tracking |
| Grafana          | http://localhost:3000 | Dashboards          |
| Prometheus       | http://localhost:9090 | Metrics             |

## Configuration

Configuration uses a 3-layer approach:

1. **`config/defaults.env`** - Sensible defaults (committed, works out of box)
2. **`.env`** - Local overrides (gitignored)
3. **Environment variables** - Runtime overrides

Key settings:

```bash
# Classifier selection
TRIAGE_CLASSIFIER_VERSION=v4

# LLM provider (default: local LM Studio)
TRIAGE_LANGUAGE_MODEL=lm_studio/gemma-3-12b-it-qat

# Or use OpenAI
# TRIAGE_LANGUAGE_MODEL=openai/gpt-4o-mini
# OPENAI_API_KEY=sk-...
```

## Requirements

- **Python** 3.11+
- **Docker** and Docker Compose
- **uv** (recommended) or pip
- **LM Studio** (for local models) or OpenAI API key

## Development

```bash
# Run tests
make test

# Run with coverage
make test-coverage

# Lint code
make lint

# Auto-fix lint issues
make lint-fix

# Full reset (removes all data)
make full-reset
```

## Documentation

- [System Architecture](docs/system-architecture.md)
- [Agent Capabilities](docs/agents-overview.md)
- [Multi-Agent Communication](docs/multi-agent-communication.md)
- [Building Agents](docs/building-agents-eggai.md)
- [RAG with Vespa](docs/agentic-rag.md)
- [Classifier Guide](docs/advanced-topics/agent-optimization.md)
- [Deployment Guide](docs/advanced-topics/multi-environment-deployment.md)

## License

MIT
