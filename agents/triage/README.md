# Triage Agent

ML-powered message classification and routing to specialized agents.

## What it does

- Classifies incoming messages using ML
- Routes to: Billing, Claims, Policies, or Escalation agents
- Handles small talk directly
- Supports 9 classifier versions (v0-v8)

## Structure

```
triage/
├── classifiers/           # All classifier implementations
│   ├── v0/               # Minimal DSPy prompt
│   ├── v1/               # Enhanced DSPy prompt
│   ├── v2/               # COPRO optimized few-shot
│   ├── v3/               # Few-shot with MLflow
│   ├── v4/               # Zero-shot COPRO (default)
│   ├── v5/               # Attention-based neural
│   ├── v6/               # OpenAI fine-tuned
│   ├── v7/               # Gemma3 LoRA fine-tuned
│   ├── v8/               # RoBERTa LoRA fine-tuned
│   ├── base.py           # Unified interface
│   └── registry.py       # Classifier factory
├── evaluation/            # Evaluation framework
├── data_sets/            # Training/testing data
├── agent.py              # Main agent
└── config.py             # Configuration
```

## Quick Usage

```python
from agents.triage.classifiers import get_classifier, list_classifiers

# Get a specific classifier
classifier = get_classifier("v4")
result = classifier.classify("User: What's my bill?")
print(result.target_agent)  # BillingAgent

# List all available classifiers
for info in list_classifiers():
    print(f"{info.version}: {info.name}")
```

## Configuration

```bash
TRIAGE_LANGUAGE_MODEL=lm_studio/gemma-3-12b-it  # Or openai/gpt-4o-mini
TRIAGE_CLASSIFIER_VERSION=v4  # v0-v8 available
```

## Classifier Versions

| Version | Name              | Type  | Training Required |
| ------- | ----------------- | ----- | ----------------- |
| v0      | Minimal Prompt    | LLM   | No                |
| v1      | Enhanced Prompt   | LLM   | No                |
| v2      | COPRO Optimized   | LLM   | One-time          |
| v3      | Few-Shot MLflow   | Local | Yes               |
| v4      | Zero-Shot COPRO   | LLM   | One-time          |
| v5      | Attention Network | Local | Yes               |
| v6      | OpenAI Fine-tuned | API   | Yes               |
| v7      | Gemma Fine-tuned  | Local | Yes               |
| v8      | RoBERTa LoRA      | Local | Yes               |

## Training

```bash
make train-v3  # Few-shot baseline
make train-v5  # Attention network
make train-v6  # OpenAI fine-tuning
make train-v7  # Gemma LoRA fine-tuning
make train-v8  # RoBERTa LoRA fine-tuning
```

## Testing

```bash
make test                    # Run all tests
make benchmark-classifiers   # Benchmark all classifiers
```
