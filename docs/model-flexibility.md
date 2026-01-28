# Model Flexibility

The system supports multiple LLM providers through LiteLLM, enabling deployment
across cloud and on-premises environments.

## Overview

LiteLLM provides a unified interface to 100+ LLM providers:

- **OpenAI** (GPT-4, GPT-4o, GPT-4o-mini)
- **Azure OpenAI**
- **Anthropic** (Claude)
- **Local Models** (LM Studio, Ollama)
- **Open Source** (Gemma, Llama, Mistral)

## Configuration

### Environment Variables

```bash
# OpenAI
LLM_MODEL=openai/gpt-4o-mini
OPENAI_API_KEY=sk-...

# Azure OpenAI
LLM_MODEL=azure/gpt-4
AZURE_API_KEY=...
AZURE_API_BASE=https://your-resource.openai.azure.com/

# Local LM Studio
LLM_MODEL=lm_studio/local-model
LM_STUDIO_API_BASE=http://localhost:1234/v1

# Anthropic Claude
LLM_MODEL=anthropic/claude-3-sonnet
ANTHROPIC_API_KEY=...
```

### Code Configuration

```python
from libraries.ml.dspy.language_model import dspy_set_language_model

# Configure at startup
dspy_set_language_model(settings)
```

## Provider Setup

### OpenAI

```bash
# .env
LLM_MODEL=openai/gpt-4o-mini
OPENAI_API_KEY=sk-your-key-here
```

Features:

- Fastest streaming
- Best tool calling support
- Highest accuracy for complex tasks

### Azure OpenAI

```bash
# .env
LLM_MODEL=azure/gpt-4
AZURE_API_KEY=your-azure-key
AZURE_API_BASE=https://your-resource.openai.azure.com/
AZURE_API_VERSION=2024-02-15-preview
```

Features:

- Enterprise compliance (SOC2, HIPAA)
- Regional deployment
- Private endpoints

### Local LM Studio

```bash
# .env
LLM_MODEL=lm_studio/your-model-name
LM_STUDIO_API_BASE=http://localhost:1234/v1

# No API key needed for local
```

Setup:

1. Download LM Studio
2. Load a model (e.g., Mistral, Llama)
3. Start local server on port 1234
4. Configure environment variables

Features:

- No API costs
- Data stays on-premises
- Works offline

### Ollama

```bash
# .env
LLM_MODEL=ollama/llama3
OLLAMA_API_BASE=http://localhost:11434
```

Setup:

```bash
# Install Ollama
curl https://ollama.ai/install.sh | sh

# Pull a model
ollama pull llama3

# Start server (usually auto-starts)
ollama serve
```

## Model Selection by Use Case

| Use Case           | Recommended Model             | Reason                |
| ------------------ | ----------------------------- | --------------------- |
| Production (cloud) | `openai/gpt-4o-mini`          | Best cost/performance |
| High accuracy      | `openai/gpt-4o`               | Most capable          |
| Enterprise         | `azure/gpt-4`                 | Compliance            |
| On-premises        | `lm_studio/mistral-7b`        | No data egress        |
| Development        | `ollama/llama3`               | Free, fast iteration  |
| Cost-sensitive     | `openai/gpt-4o-mini` or local | Lowest per-token      |

## DSPy Integration

The language model integrates with DSPy for all agent operations:

```python
import dspy
from libraries.ml.dspy.language_model import get_language_model

# Get configured model
lm = get_language_model(settings)

# Use with DSPy
dspy.configure(lm=lm)

# Now all DSPy modules use this model
class MyModule(dspy.Module):
    def forward(self, query):
        return dspy.ChainOfThought("query -> answer")(query=query)
```

## Streaming Support

All providers support streaming through the same interface:

```python
# Streaming works identically regardless of provider
chunks = dspy.streamify(model, async_streaming=True)(input=query)

async for chunk in chunks:
    yield chunk
```

## Fallback Configuration

Configure fallback models for reliability:

```python
# In settings
class Settings(BaseAgentConfig):
    llm_model: str = "openai/gpt-4o-mini"
    llm_fallback_model: str = "azure/gpt-4"

# In code
try:
    response = await primary_model.generate(prompt)
except Exception:
    response = await fallback_model.generate(prompt)
```

## Cost Optimization

### Token Pricing (as of 2024)

| Model           | Input (per 1M) | Output (per 1M) |
| --------------- | -------------- | --------------- |
| gpt-4o-mini     | $0.15          | $0.60           |
| gpt-4o          | $5.00          | $15.00          |
| gpt-4-turbo     | $10.00         | $30.00          |
| claude-3-sonnet | $3.00          | $15.00          |
| Local models    | $0             | $0              |

### Cost Reduction Strategies

1. **Use smaller models for routing**: Triage can use gpt-4o-mini
2. **Local classifiers**: v5-v8 classifiers use no API
3. **Cache common queries**: Reduce repeated API calls
4. **Batch where possible**: Reduce per-request overhead

## Testing Different Models

```bash
# Quick model comparison
LLM_MODEL=openai/gpt-4o-mini python -m agents.billing.main &
# Test queries...

LLM_MODEL=ollama/llama3 python -m agents.billing.main &
# Compare responses...
```

## Key Files

```
libraries/ml/dspy/
├── language_model.py    # LiteLLM configuration
└── optimizer.py         # Model for optimization

config/
└── .env.example         # Environment template
```

## Troubleshooting

### Connection Errors

```bash
# Check API endpoint
curl -I $LM_STUDIO_API_BASE/v1/models

# Verify API key
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY"
```

### Model Not Found

```bash
# List available Ollama models
ollama list

# List LM Studio models
curl $LM_STUDIO_API_BASE/v1/models
```

### Slow Responses

- Check model size (smaller = faster)
- Verify GPU acceleration is enabled
- Consider streaming for perceived latency

---

**Previous:** [Streaming Architecture](streaming-architecture.md) | **Next:**
[Agentic RAG](agentic-rag.md) | [Back to Index](README.md)
