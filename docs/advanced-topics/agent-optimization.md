# Agent Performance Optimization

This guide covers optimization techniques for both DSPy-based agents and the Triage classifier.

## DSPy Agent Optimization

### SIMBA Optimizer

SIMBA (Simple Instruction Following via Bootstrapping) optimizes agent prompts automatically. Agents with optimizer scripts:

```bash
uv run -m agents.billing.dspy_modules.billing_optimizer_simba
uv run -m agents.claims.dspy_modules.claims_optimizer_simba
uv run -m agents.policies.agent.optimization.policies_optimizer_simba
```

Optimized prompts are saved as JSON files (e.g., `optimized_billing_simba.json`) and loaded automatically at startup.

## Triage Classifier Optimization

The Triage agent evolved through multiple versions, from basic prompts to advanced neural networks:

### Classifier Evolution

| Version | Type | Optimization Method |
|---------|------|---------------------|
| **v0** | Basic prompt | None (baseline) |
| **v1** | Enhanced prompt | Manual examples |
| **v2** | DSPy few-shot | Auto-optimized with examples |
| **v3** | ML baseline | Logistic regression training |
| **v4** | DSPy zero-shot | COPRO optimization (recommended) |
| **v5** | Neural network | Attention-based training |
| **v6** | Fine-tuned LLM | OpenAI GPT-4o-mini fine-tuning |
| **v7** | Local LLM | Gemma3 with LoRA fine-tuning |
| **v8** | Local transformer | RoBERTa with LoRA fine-tuning |

### 1. Unoptimized Baselines (v0, v1)

No setup required — these use LLM prompts directly. Test via:

```bash
TRIAGE_CLASSIFIER_VERSION=v0 make test
TRIAGE_CLASSIFIER_VERSION=v1 make test
```

### 2. DSPy-Based Optimizers (v2, v4)

Optimize prompts with DSPy, then test:

```bash
# Optimize few-shot classifier
uv run -m agents.triage.classifiers.v2.classifier_v2_optimizer

# Optimize zero-shot COPRO classifier (recommended)
uv run -m agents.triage.classifiers.v4.classifier_v4_optimizer
```

**COPRO** (Compositional Preference Optimization) automatically discovers optimal prompts without training data.

### 3. Custom ML Classifiers (v3, v5)

For improved latency with no API calls:

```bash
# Train classifiers
make train-v3  # Few-shot classifier
make train-v5  # Attention-based neural network
```

### 4. Fine-tuned Models (v6, v7, v8)

For production deployments with high accuracy:

```bash
# v6: OpenAI fine-tuning (requires API key)
make train-v6

# v7: Local Gemma3 with LoRA (no API calls)
make train-v7

# v8: Local RoBERTa with LoRA (fastest inference)
uv run -m agents.triage.classifiers.v8.finetune_trainer
```

**v7 and v8** run entirely locally, making them ideal for privacy-sensitive deployments.

### 5. Performance Comparison

Benchmark all classifiers:

```bash
make benchmark-classifiers

# View results in MLflow UI
# http://localhost:5001
```

### 6. Performance Monitoring

View metrics and comparisons in MLflow:
- **URL**: http://localhost:5001
- **Experiments**: Organized by classifier version
- **Metrics**: Accuracy, F1 score, latency (mean/p95/max), token usage
- **Reports**: HTML reports in `agents/triage/tests/reports/`

#### Key Metrics to Compare:
- **Accuracy**: Classification correctness
- **Latency**: Response time (critical for user experience)
- **Token Usage**: Cost implications (prompt + completion tokens)
- **Consistency**: Performance variance across test runs

## Optimization Journey: From Basic to Advanced

### The Evolution Story

1. **v0 → v1**: Added manual examples to improve consistency
2. **v1 → v2**: Automated optimization with DSPy few-shot learning
3. **v2 → v4**: Eliminated need for examples with zero-shot COPRO
4. **Parallel Track**: v3 (traditional ML) and v5 (neural) for production efficiency
5. **v6**: Cloud-based fine-tuning via OpenAI for maximum accuracy
6. **v7/v8**: Local fine-tuned models for privacy and cost efficiency

### Optimization Strategies

#### When to Use Each Approach:

1. **SIMBA (Individual Agents)**: Best for optimizing ReAct agents with complex tool usage
2. **COPRO (Triage v4)**: Zero-shot prompt optimization when you lack training data
3. **Few-Shot (Triage v2)**: When you have limited examples but want DSPy flexibility
4. **Custom ML (v3/v5)**: Production deployments requiring low latency and cost

#### Trade-offs:

| Approach | Pros | Cons |
|----------|------|------|
| **Unoptimized (v0/v1)** | Simple, no setup | Lower accuracy, inconsistent |
| **DSPy Optimized (v2/v4)** | Auto-optimization, flexible | Requires compilation step |
| **Custom ML (v3/v5)** | Fast inference, low cost | Requires training data, less flexible |
| **Cloud Fine-tuned (v6)** | High accuracy, simple deployment | API costs, data sent to cloud |
| **Local Fine-tuned (v7/v8)** | No API costs, data privacy | Requires local GPU, more setup |


---

**Previous:** [Vespa Search Guide](../vespa-search-guide.md) | **Next:** [Deployment Guide](multi-environment-deployment.md)
