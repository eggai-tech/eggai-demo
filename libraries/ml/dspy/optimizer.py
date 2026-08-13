import datetime
import os
import time
from collections.abc import Callable

import dspy
import mlflow

# Bound to its own name: mlflow is py.typed and does not re-export the dspy
# submodule, so reaching it as an attribute of the package does not type-check.
import mlflow.dspy as mlflow_dspy
from dspy.evaluate import Evaluate

from libraries.observability.logger import get_console_logger

logger = get_console_logger("dspy_simba.optimizer")


class SIMBAOptimizer:
    def __init__(
        self,
        metric: Callable,
        max_steps: int = 8,
        max_demos: int = 5,
        verbose: bool = True,
    ):
        self.metric = metric
        self.max_steps = max_steps
        self.max_demos = max_demos
        self.verbose = verbose
        self.dspy_optimizer = dspy.SIMBA(
            metric=metric, max_steps=max_steps, max_demos=max_demos
        )

    def optimize(
        self,
        program: dspy.Module,
        trainset: list[dspy.Example],
        devset: list[dspy.Example] | None = None,
        experiment_name: str = "dspy_optimization",
        run_name: str | None = None,
        output_path: str | None = None,
        seed: int = 42,
    ) -> dspy.Module:
        mlflow.set_experiment(experiment_name)

        if run_name is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"simba_optimization_{timestamp}"

        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(
                {
                    "optimizer": "SIMBA",
                    "max_steps": self.max_steps,
                    "max_demos": self.max_demos,
                    "train_examples": len(trainset),
                    "dev_examples": len(devset) if devset else 0,
                    "seed": seed,
                }
            )

            mlflow_dspy.autolog(
                log_compiles=True,
                log_traces=True,
                log_evals=True,
                log_traces_from_compile=True,
                log_traces_from_eval=True,
            )

            evaluator = (
                Evaluate(
                    devset=devset, metric=self.metric, num_threads=min(4, len(devset))
                )
                if devset
                else None
            )

            if evaluator is not None:
                logger.info("Evaluating baseline performance...")
                # Evaluate returns an EvaluationResult, not a float; .score is the
                # number. The object supports neither arithmetic nor format specs.
                baseline_score = evaluator(program).score
                logger.info(f"Baseline score: {baseline_score:.3f}")
                mlflow.log_metric("baseline_score", baseline_score)
            else:
                logger.info("No development set provided, skipping baseline evaluation")
                baseline_score = None

            logger.info(f"Starting SIMBA optimization with {self.max_steps} steps...")
            start_time = time.time()

            optimized_program = self.dspy_optimizer.compile(
                program, trainset=trainset, seed=seed
            )

            optimization_time = time.time() - start_time
            logger.info(f"Optimization completed in {optimization_time:.1f} seconds")
            mlflow.log_metric("optimization_time_seconds", optimization_time)

            if evaluator is not None:
                logger.info("Evaluating optimized performance...")
                optimized_score = evaluator(optimized_program).score
                logger.info(f"Optimized score: {optimized_score:.3f}")

                if baseline_score is not None:
                    improvement = optimized_score - baseline_score
                    relative_improvement = (
                        improvement / max(baseline_score, 1e-9)
                    ) * 100
                    logger.info(f"Absolute improvement: {improvement:.3f}")
                    logger.info(f"Relative improvement: {relative_improvement:.1f}%")

                    mlflow.log_metric("optimized_score", optimized_score)
                    mlflow.log_metric("absolute_improvement", improvement)
                    mlflow.log_metric("relative_improvement", relative_improvement)

            if output_path:
                self._save_optimized_program(optimized_program, output_path)
                mlflow.log_artifact(output_path)
                logger.info(f"Saved optimized program to {output_path}")

            return optimized_program

    def _save_optimized_program(self, program: dspy.Module, output_path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        program.save(output_path)


def optimize_react_agent(
    agent_class: type,
    signature_class: type,
    tools: list[Callable],
    trainset: list[dspy.Example],
    devset: list[dspy.Example] | None = None,
    metric: Callable | None = None,
    max_steps: int = 8,
    max_demos: int = 5,
    max_iters: int = 5,
    output_path: str | None = None,
    experiment_name: str = "react_optimization",
    seed: int = 42,
) -> dspy.Module:
    agent = agent_class(signature_class, tools=tools, max_iters=max_iters)

    if metric is None:

        def default_metric(example, prediction, trace=None) -> float:
            gold = getattr(example, "answer", "")
            pred = getattr(prediction, "answer", "")

            if not gold or not pred:
                return 0.0

            return float(gold.lower().strip() == pred.lower().strip())

        metric = default_metric

    optimizer = SIMBAOptimizer(metric=metric, max_steps=max_steps, max_demos=max_demos)

    return optimizer.optimize(
        program=agent,
        trainset=trainset,
        devset=devset,
        experiment_name=experiment_name,
        output_path=output_path,
        seed=seed,
    )


def load_optimized_react_agent(
    path: str,
    agent_class: type,
    signature_class: type,
    tools: list[Callable],
    max_iters: int = 5,
) -> dspy.Module:
    if os.path.exists(path):
        try:
            logger.info(f"Loading optimized agent from {path}")
            # _save_optimized_program uses Module.save(save_program=False), which
            # writes state only. State has to be loaded into a constructed module;
            # dspy.load() is for whole-program directories saved with
            # save_program=True, so it is not the counterpart here.
            agent = agent_class(signature_class, tools=tools, max_iters=max_iters)
            agent.load(path)
            return agent
        except Exception as e:
            logger.error(
                f"Failed to load optimized agent: {e}. Creating unoptimized agent instead."
            )
    else:
        logger.info(f"No optimized agent found at {path}. Creating unoptimized agent.")

    return agent_class(signature_class, tools=tools, max_iters=max_iters)
