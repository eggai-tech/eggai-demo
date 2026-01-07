# Multi-Agent Insurance Support System

[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge&logo=opensourceinitiative&logoColor=white)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![EggAI SDK](https://img.shields.io/badge/Built%20with-EggAI-orange?style=for-the-badge)](https://github.com/eggai-tech/eggai)

A production-ready multi-agent system built with [EggAI](https://github.com/eggai-tech/eggai) where AI agents collaborate to provide personalized insurance support. Features billing inquiries, claims processing, policy information retrieval (RAG), and intelligent routing.

![Chat UI Screenshot](https://raw.githubusercontent.com/eggai-tech/EggAI/refs/heads/main/docs/docs/assets/support-chat.png)

> **Note:** Runs completely locally with LM Studio - no cloud services or API keys required!

## Quick Start

### Prerequisites

- **Python** 3.11+
- **Docker** and **Docker Compose**
- **LM Studio** (for local models) or OpenAI API key (for cloud models)

### 1. Setup

```bash
# Clone the repository
git clone git@github.com:eggai-tech/eggai-demo.git
cd eggai-demo

# Create virtual environment and install dependencies
make setup
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Copy environment configuration
cp .env.example .env

# (Optional) Configure Guardrails for content moderation
# guardrails configure --token $GUARDRAILS_TOKEN
# guardrails hub install hub://guardrails/toxic_language
```

### 2. Configure Language Models

#### Option A: Local Models (Default - No API Keys Required)

1. Download and install [LM Studio](https://lmstudio.ai/)
2. Launch LM Studio and load a compatible model (e.g., `google/gemma-3-4b`)
3. Start the local server (should run on http://localhost:1234)

#### Option B: OpenAI Models

Edit `.env` and uncomment the OpenAI model lines for every agent:

```bash
# Uncomment these lines in .env
# TRIAGE_LANGUAGE_MODEL_API_BASE=https://api.openai.com/v1
# TRIAGE_LANGUAGE_MODEL=openai/gpt-4o-mini
# OPENAI_API_KEY=your-api-key-here
[...]
```

### 3. Start Platform Services

```bash
make docker-up  # Start all required services (Kafka, Vespa, Temporal, etc.)
```

All services are accessible directly from the chat UI header, or you can open them individually via their URLs:

| Service     | Local URL                            | Description                            | Product URL                              |
|-------------|--------------------------------------|----------------------------------------|-------------------------------------------|
| Redpanda    | [http://localhost:9878](http://localhost:9878)   | Kafka-compatible message queue         | [redpanda.com](https://redpanda.com)       |
| Vespa       | [http://localhost:9894](http://localhost:9894)   | Vector search engine and ranking       | [vespa.ai](https://vespa.ai)               |
| Temporal    | [http://localhost:9892](http://localhost:9892)   | Orchestration engine for workflows     | [temporal.io](https://temporal.io)         |
| MLflow      | [http://localhost:9889](http://localhost:9889)   | Machine learning experiment tracking   | [mlflow.org](https://mlflow.org)           |
| Grafana     | [http://localhost:9884](http://localhost:9884)   | Visualization and dashboarding tool    | [grafana.com](https://grafana.com)         |
| Prometheus  | [http://localhost:9883](http://localhost:9883)   | Metrics collection and time-series DB  | [prometheus.io](https://prometheus.io)     |

## Run the System

```bash
make start-all
```

**Open http://localhost:9903 to start chatting!**

The chat interface includes example questions to get started:

- **Support categories** with clickable example questions
- **Policy Inquiries** - Coverage and policy details
- **Billing & Payments** - Premiums and payment info  
- **Claims Support** - File claims and check status
- **General Support** - Escalations and other help

### Usage

You can also interact with the system using free-form natural language.  
Simply type your request into the chat input at the bottom of the interface.

Here are some example prompts you can try:

- _"What's my premium for policy B67890?"_
- _"I want to file a claim"_
- _"What does my home insurance cover?"_
- _"I have a complaint about my service"_

The system will automatically route your request to the appropriate agent.

## Documentation

- [System Architecture](docs/system-architecture.md)
- [Agent Capabilities Overview](docs/agents-overview.md)
- [Multi-Agent Communication](docs/multi-agent-communication.md)
- [Building Agents Guide](docs/building-agents-eggai.md)
- [Document Ingestion with Temporal](docs/ingestion-pipeline.md)
- [RAG with Vespa](docs/agentic-rag.md)
- [Vespa Search Guide](docs/vespa-search-guide.md)
- [Agent & Prompt Optimization](docs/advanced-topics/agent-optimization.md)
- [Deployment Guide](docs/advanced-topics/multi-environment-deployment.md)

## Development

### Testing

```bash
# Unit tests (no external dependencies - runs in CI)
make test-ci

# Integration tests (requires docker-compose infrastructure)
docker compose up -d
make test-integration

# All tests
make test-all
```

### Code Quality

```bash
make lint        # Check code quality
make lint-fix    # Auto-fix lint issues
```
