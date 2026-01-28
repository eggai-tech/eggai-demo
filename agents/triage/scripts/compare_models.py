#!/usr/bin/env python3
"""
Compare classifier performance across different models and versions.

Models tested:
- OpenAI gpt-4o-mini
- LM Studio Gemma 3 12B
- LM Studio Mistral Nemo 12B

Classifier versions tested:
- v0: Minimal Prompt (unoptimized baseline)
- v1: Enhanced Prompt (unoptimized with better descriptions)
- v4: Zero-Shot COPRO (optimized instructions)
- v2: COPRO Optimized (few-shot examples)

Usage:
    python -m agents.triage.scripts.compare_models
    python -m agents.triage.scripts.compare_models --models openai gemma
    python -m agents.triage.scripts.compare_models --versions v0 v4
"""

import argparse
import importlib
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import dspy
import mlflow
from dotenv import load_dotenv
from tabulate import tabulate

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from agents.triage.models import TargetAgent
from libraries.ml.dspy.language_model import TrackingLM
from libraries.observability.logger import get_console_logger

load_dotenv()

logger = get_console_logger("compare_models")


@dataclass
class ModelConfig:
    name: str
    model_id: str
    api_base: str | None = None
    requires_api_key: bool = False


@dataclass
class TestResult:
    model: str
    version: str
    input_text: str
    expected: str
    predicted: str
    correct: bool
    latency_ms: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    error: str | None = None


@dataclass
class AggregateResult:
    model: str
    version: str
    accuracy: float
    avg_latency_ms: float
    p95_latency_ms: float
    total_tokens: int
    avg_tokens: float
    error_count: int
    total_tests: int


# Model configurations
MODELS = {
    "openai": ModelConfig(
        name="OpenAI GPT-4o-mini",
        model_id="openai/gpt-4o-mini",
        api_base=None,
        requires_api_key=True,
    ),
    "gemma": ModelConfig(
        name="Gemma 3 12B QAT",
        model_id="lm_studio/gemma-3-12b-it-qat",
        api_base="http://localhost:1234/v1/",
    ),
    "mistral": ModelConfig(
        name="Mistral Nemo 12B",
        model_id="lm_studio/mistral-nemo-instruct-2407",
        api_base="http://localhost:1234/v1/",
    ),
}

# Classifier versions to test
VERSIONS = ["v0", "v1", "v2", "v4"]

# Test cases with expected agent
# Available agents: BillingAgent, PolicyAgent, ClaimsAgent, EscalationAgent, ChattyAgent
TEST_CASES = [
    ("User: I need help with my insurance claim", TargetAgent.ClaimsAgent),
    ("User: My claim was denied, can you explain why?", TargetAgent.ClaimsAgent),
    ("User: I had a car accident and need to file a claim", TargetAgent.ClaimsAgent),
    ("User: What's covered under my policy?", TargetAgent.PolicyAgent),
    ("User: I want to add my spouse to my policy", TargetAgent.PolicyAgent),
    ("User: Can you explain my deductible?", TargetAgent.PolicyAgent),
    ("User: How do I pay my premium?", TargetAgent.BillingAgent),
    ("User: When is my next payment due?", TargetAgent.BillingAgent),
    ("User: I need a copy of my invoice", TargetAgent.BillingAgent),
    ("User: Hello, how are you today?", TargetAgent.ChattyAgent),
    ("User: What's the weather like?", TargetAgent.ChattyAgent),
    ("User: Tell me a joke", TargetAgent.ChattyAgent),
]


def configure_model(config: ModelConfig) -> TrackingLM:
    """Configure DSPy with the specified model."""
    if config.requires_api_key and not os.getenv("OPENAI_API_KEY"):
        raise ValueError(f"OPENAI_API_KEY required for {config.name}")

    lm = TrackingLM(
        config.model_id,
        cache=False,
        api_base=config.api_base,
    )
    dspy.configure(lm=lm)
    return lm


def get_classifier(version: str):
    """Dynamically import and return classifier function."""
    # Clear cached imports to ensure fresh model config
    module_name = f"agents.triage.classifiers.{version}"
    if version in ["v2", "v4"]:
        module_name = f"agents.triage.classifiers.{version}.classifier_{version}"

    # Remove from cache if exists
    if module_name in sys.modules:
        del sys.modules[module_name]

    module = importlib.import_module(module_name)
    return getattr(module, f"classifier_{version}")


def run_single_test(
    classifier_fn, chat_history: str, expected: TargetAgent, model_name: str, version: str
) -> TestResult:
    """Run a single classification test."""
    try:
        start_time = time.perf_counter()
        result = classifier_fn(chat_history=chat_history)
        latency_ms = (time.perf_counter() - start_time) * 1000

        predicted = result.target_agent
        correct = predicted == expected

        metrics = getattr(result, "metrics", None)
        prompt_tokens = metrics.prompt_tokens if metrics else 0
        completion_tokens = metrics.completion_tokens if metrics else 0
        total_tokens = metrics.total_tokens if metrics else 0

        return TestResult(
            model=model_name,
            version=version,
            input_text=chat_history[:50] + "..." if len(chat_history) > 50 else chat_history,
            expected=expected.name,
            predicted=predicted.name if hasattr(predicted, "name") else str(predicted),
            correct=correct,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )
    except Exception as e:
        return TestResult(
            model=model_name,
            version=version,
            input_text=chat_history[:50] + "...",
            expected=expected.name,
            predicted="ERROR",
            correct=False,
            latency_ms=0,
            error=str(e),
        )


def aggregate_results(results: list[TestResult], model: str, version: str) -> AggregateResult:
    """Aggregate results for a model/version combination."""
    filtered = [r for r in results if r.model == model and r.version == version]
    if not filtered:
        return AggregateResult(
            model=model, version=version, accuracy=0, avg_latency_ms=0,
            p95_latency_ms=0, total_tokens=0, avg_tokens=0, error_count=0, total_tests=0
        )

    correct = sum(1 for r in filtered if r.correct)
    errors = sum(1 for r in filtered if r.error)
    latencies = [r.latency_ms for r in filtered if not r.error]
    tokens = [r.total_tokens for r in filtered if not r.error]

    latencies_sorted = sorted(latencies) if latencies else [0]
    p95_idx = int(len(latencies_sorted) * 0.95)

    return AggregateResult(
        model=model,
        version=version,
        accuracy=(correct / len(filtered)) * 100 if filtered else 0,
        avg_latency_ms=sum(latencies) / len(latencies) if latencies else 0,
        p95_latency_ms=latencies_sorted[min(p95_idx, len(latencies_sorted) - 1)],
        total_tokens=sum(tokens),
        avg_tokens=sum(tokens) / len(tokens) if tokens else 0,
        error_count=errors,
        total_tests=len(filtered),
    )


def print_comparison_table(aggregates: list[AggregateResult]):
    """Print formatted comparison table."""
    headers = [
        "Model", "Version", "Accuracy", "Avg Latency", "P95 Latency",
        "Avg Tokens", "Errors"
    ]
    rows = []
    for a in aggregates:
        rows.append([
            a.model,
            a.version,
            f"{a.accuracy:.1f}%",
            f"{a.avg_latency_ms:.0f}ms",
            f"{a.p95_latency_ms:.0f}ms",
            f"{a.avg_tokens:.0f}",
            a.error_count,
        ])

    print("\n" + "=" * 80)
    print("CLASSIFIER COMPARISON RESULTS")
    print("=" * 80)
    print(tabulate(rows, headers=headers, tablefmt="grid"))
    print()


def print_detailed_results(results: list[TestResult]):
    """Print detailed per-test results."""
    headers = ["Model", "Ver", "Input", "Expected", "Predicted", "✓", "Latency"]
    rows = []
    for r in results:
        rows.append([
            r.model[:15],
            r.version,
            r.input_text[:30] + "...",
            r.expected,
            r.predicted,
            "✓" if r.correct else "✗",
            f"{r.latency_ms:.0f}ms" if not r.error else "ERR",
        ])

    print("\n" + "-" * 80)
    print("DETAILED RESULTS")
    print("-" * 80)
    print(tabulate(rows, headers=headers, tablefmt="simple"))


def log_to_mlflow(aggregates: list[AggregateResult], experiment_name: str = "classifier_comparison"):
    """Log comparison results to MLflow."""
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001"))
    mlflow.set_experiment(experiment_name)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for agg in aggregates:
        run_name = f"{agg.model}_{agg.version}_{timestamp}"
        with mlflow.start_run(run_name=run_name):
            # Log parameters
            mlflow.log_param("model", agg.model)
            mlflow.log_param("classifier_version", agg.version)
            mlflow.log_param("test_count", agg.total_tests)
            mlflow.log_param("is_optimized", agg.version in ["v2", "v4"])

            # Log metrics
            mlflow.log_metric("accuracy", agg.accuracy)
            mlflow.log_metric("avg_latency_ms", agg.avg_latency_ms)
            mlflow.log_metric("p95_latency_ms", agg.p95_latency_ms)
            mlflow.log_metric("avg_tokens", agg.avg_tokens)
            mlflow.log_metric("total_tokens", agg.total_tokens)
            mlflow.log_metric("error_count", agg.error_count)

            # Add tags for filtering
            mlflow.set_tag("model_type", "cloud" if "OpenAI" in agg.model else "local")
            mlflow.set_tag("optimization_type", "optimized" if agg.version in ["v2", "v4"] else "unoptimized")

    logger.info(f"Logged {len(aggregates)} runs to MLflow experiment: {experiment_name}")


def main():
    parser = argparse.ArgumentParser(description="Compare classifier models and versions")
    parser.add_argument(
        "--models", nargs="+", default=list(MODELS.keys()),
        choices=list(MODELS.keys()),
        help="Models to test"
    )
    parser.add_argument(
        "--versions", nargs="+", default=VERSIONS,
        help="Classifier versions to test"
    )
    parser.add_argument(
        "--detailed", action="store_true",
        help="Show detailed per-test results"
    )
    parser.add_argument(
        "--mlflow", action="store_true",
        help="Log results to MLflow"
    )
    args = parser.parse_args()

    all_results: list[TestResult] = []
    aggregates: list[AggregateResult] = []

    print("\n" + "=" * 80)
    print("CLASSIFIER MODEL COMPARISON")
    print(f"Models: {', '.join(args.models)}")
    print(f"Versions: {', '.join(args.versions)}")
    print(f"Test cases: {len(TEST_CASES)}")
    print("=" * 80 + "\n")

    for model_key in args.models:
        config = MODELS[model_key]
        logger.info(f"\nTesting model: {config.name}")

        try:
            lm = configure_model(config)
            logger.info(f"  Configured: {config.model_id}")
        except Exception as e:
            logger.error(f"  Failed to configure model: {e}")
            continue

        for version in args.versions:
            logger.info(f"  Testing classifier {version}...")

            try:
                classifier = get_classifier(version)
            except Exception as e:
                logger.error(f"    Failed to load classifier {version}: {e}")
                continue

            for chat_history, expected in TEST_CASES:
                result = run_single_test(
                    classifier, chat_history, expected, config.name, version
                )
                all_results.append(result)

                status = "✓" if result.correct else "✗"
                if result.error:
                    status = f"ERR: {result.error[:30]}"
                logger.debug(f"    {status} {chat_history[:40]}...")

            agg = aggregate_results(all_results, config.name, version)
            aggregates.append(agg)
            logger.info(
                f"    {version}: {agg.accuracy:.1f}% accuracy, "
                f"{agg.avg_latency_ms:.0f}ms avg latency"
            )

    # Print results
    print_comparison_table(aggregates)

    if args.detailed:
        print_detailed_results(all_results)

    # Log to MLflow
    if args.mlflow and aggregates:
        log_to_mlflow(aggregates)
        print(f"\n✓ Results logged to MLflow: http://localhost:5001")

    # Summary
    print("\nSUMMARY:")
    if aggregates:
        best_accuracy = max(aggregates, key=lambda x: x.accuracy)
        best_latency = min(
            (a for a in aggregates if a.avg_latency_ms > 0),
            key=lambda x: x.avg_latency_ms,
            default=None
        )
        print(f"  Best accuracy: {best_accuracy.model} {best_accuracy.version} ({best_accuracy.accuracy:.1f}%)")
        if best_latency:
            print(f"  Fastest: {best_latency.model} {best_latency.version} ({best_latency.avg_latency_ms:.0f}ms)")


if __name__ == "__main__":
    main()
