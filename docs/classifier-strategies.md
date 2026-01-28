# Triage Classifier Strategies

The Triage Agent uses ML-based classification to route user queries to
specialized agents. The system includes 9 classifier strategies demonstrating
evolution from basic prompting to fine-tuned local models.

## Overview

| Version | Approach                  | Latency  | LLM API | Training |
| ------- | ------------------------- | -------- | ------- | -------- |
| v0      | Basic DSPy prompt         | ~500ms   | Yes     | No       |
| v1      | Enhanced prompt           | ~600ms   | Yes     | No       |
| v2      | COPRO optimized           | ~500ms   | Yes     | One-time |
| v3      | Few-shot (local embeddings) | ~50ms  | No      | Yes      |
| **v4**  | Zero-shot COPRO           | ~400ms   | Yes     | One-time |
| v5      | Attention network         | ~20ms    | No      | Yes      |
| v6      | OpenAI fine-tuned         | ~300ms   | Yes     | Yes      |
| v7      | Gemma fine-tuned          | ~100ms   | No      | Yes      |
| v8      | RoBERTa LoRA              | ~50ms    | No      | Yes      |

*Latency values are approximate estimates.*

**Default:** v4 (Zero-shot COPRO) - best balance of accuracy and latency

## Configuration

```bash
# Set classifier version via environment variable
TRIAGE_CLASSIFIER_VERSION=v4

# Or in code
from agents.triage.classifiers import get_classifier
classifier = get_classifier("v4")
```

## Classifier Details

### v0-v2: Prompt-Based (Cloud LLM)

**v0 - Basic DSPy Prompt**

- Simple prompt asking LLM to classify intent
- No optimization, baseline performance
- Requires external API call

**v1 - Enhanced Prompt**

- Improved prompt with better category descriptions
- More context about each agent's capabilities
- Still requires API call

**v2 - COPRO Optimized**

- Uses DSPy COPRO optimizer to improve prompt
- One-time optimization, then fixed prompt
- Better accuracy than v0/v1

### v3: Few-Shot Local

**v3 - Few-Shot with Local Embeddings**

- Uses sentence-transformer embeddings for similarity matching
- No LLM API calls - runs entirely locally
- Models stored in MLflow registry for versioning
- Fast inference (~50ms) because it's just vector similarity
- Requires training dataset with labeled examples

### v4: Zero-Shot Optimized (Default)

**v4 - Zero-Shot COPRO**

- Best balance of accuracy and simplicity
- Optimized zero-shot prompt
- Recommended starting point
- Works well across diverse queries

### v5-v8: Fine-Tuned Models

**v5 - Attention Network**

- Custom attention-based classifier
- Ultra-fast inference (~20ms)
- Requires training on labeled data

**v6 - OpenAI Fine-Tuned**

- Fine-tuned OpenAI model
- Cloud-based, requires API
- Good accuracy, moderate latency

**v7 - Gemma Fine-Tuned**

- Fine-tuned Gemma model (local)
- No external API required
- Good for on-premises deployment

**v8 - RoBERTa LoRA**

- Parameter-efficient fine-tuning
- Fast inference (~50ms)
- Small model footprint

## Unified Interface

All classifiers implement the same interface:

```python
from agents.triage.classifiers import BaseClassifier

class YourClassifier(BaseClassifier):
    async def classify(self, query: str) -> ClassificationResult:
        # Return predicted agent and confidence
        return ClassificationResult(
            agent="BillingAgent",
            confidence=0.95
        )
```

## Training Custom Classifiers

### 1. Prepare Dataset

```python
# Format: query -> agent mapping
training_data = [
    {"query": "What's my premium?", "agent": "BillingAgent"},
    {"query": "File a claim", "agent": "ClaimsAgent"},
    # ...
]
```

### 2. Run Training

```bash
# Train v7 (Gemma)
python -m agents.triage.classifiers.v7.finetune_trainer

# Train v8 (LoRA)
python -m agents.triage.classifiers.v8.training_utils
```

### 3. Deploy

Models are automatically registered in MLflow and loaded by the classifier
registry.

## Performance Comparison

### Latency vs Accuracy Trade-off

```
Accuracy
  ↑
  │     v4●  v6●
  │   v2●     v7●
  │  v1●
  │ v0●      v3● v5● v8●
  │
  └────────────────────→ Latency (ms)
       500   300   100   50   20
```

### Cost Comparison

| Classifier | API Cost    | Compute Cost         |
| ---------- | ----------- | -------------------- |
| v0-v2      | Per-request | Minimal              |
| v3         | None        | Training + MLflow    |
| v4         | Per-request | Minimal              |
| v5-v8      | None        | Training + Inference |

## Recommendations

| Scenario         | Recommended             |
| ---------------- | ----------------------- |
| Getting started  | v4 (zero-shot)          |
| Cost-sensitive   | v3, v5, v7, v8          |
| On-premises only | v7 (Gemma) or v8 (LoRA) |
| Highest accuracy | v4 or v6                |
| Lowest latency   | v5 (20ms)               |

## Key Files

```
agents/triage/classifiers/
├── base.py              # Base classifier interface
├── registry.py          # Classifier registry
├── v0/classifier.py     # Basic prompt
├── v2/classifier_v2_optimizer.py
├── v3/classifier_v3.py  # Few-shot
├── v4/classifier_v4.py  # Zero-shot COPRO
├── v5/classifier_v5.py  # Attention network
├── v6/classifier_v6.py  # OpenAI fine-tuned
├── v7/classifier_v7.py  # Gemma fine-tuned
└── v8/classifier_v8.py  # RoBERTa LoRA
```

---

**Previous:** [Multi-Agent Communication](multi-agent-communication.md) |
**Next:** [Observability](observability.md) | [Back to Index](README.md)
