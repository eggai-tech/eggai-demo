# Triage Agent

ML-powered message classification and routing to specialized agents.

## What it does
- Classifies incoming messages using ML
- Routes to: Billing, Claims, Policies, or Escalation agents
- Handles small talk directly
- Supports multiple classifier versions (v0-v8)

## Quick Start
```bash
make start-triage
```

## Configuration
```bash
TRIAGE_LANGUAGE_MODEL=lm_studio/gemma-3-12b-it  # Or openai/gpt-4o-mini
TRIAGE_CLASSIFIER_VERSION=v4  # v0-v8 available
```

## Classifier Versions
- **v0**: Basic prompt-based
- **v1**: Enhanced prompt with examples
- **v2**: DSPy optimized few-shot
- **v3**: Baseline few-shot with training
- **v4**: Zero-shot COPRO optimized (default)
- **v5**: Attention-based neural classifier
- **v6**: OpenAI fine-tuned GPT-4o-mini
- **v7**: Gemma3 with LoRA fine-tuning (local)
- **v8**: RoBERTa with LoRA fine-tuning (local)

## Training
```bash
make train-triage-classifier-v3  # Baseline
make train-triage-classifier-v5  # Attention-based
make compile-triage-classifier-v4  # Optimize DSPy
make train-triage-classifier-v6  # Fine-tuning via OpenAI API
make train-triage-classifier-v7  # Gemma3 LoRA fine-tuning locally
make train-triage-classifier-v8  # RoBERTa LoRA fine-tuning locally
```

## Testing
```bash
make test-triage-agent
make test-triage-classifier-v4  # Test specific version
```