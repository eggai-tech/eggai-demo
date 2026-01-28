import datetime
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from libraries.observability.logger import get_console_logger


def print_progress(message, progress=None, total=None):
    if progress is not None and total is not None:
        percent = min(100, int(progress / total * 100))
        bar_length = 30
        filled_length = int(bar_length * progress // total)
        bar = "█" * filled_length + "░" * (bar_length - filled_length)
        sys.stdout.write(f"\r{message}: [{bar}] {percent}% ({progress}/{total})")
        sys.stdout.flush()
    else:
        chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        spinner = chars[int(time.time() * 10) % len(chars)]
        sys.stdout.write(f"\r{spinner} {message}...")
        sys.stdout.flush()


@dataclass
class OptimizerConfig:
    agent_name: str
    experiment_name: str
    program: Any
    metric_fn: Callable
    dataset_creator: Callable
    dataset_converter: Callable
    critical_privacy_text: str
    output_filename: str
    settings: Any
    emergency_instruction: str


def _show_spinner(message: str, timeout: float):
    elapsed = 0
    while elapsed < timeout:
        chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        for char in chars:
            sys.stdout.write(f"\r{char} {message} (elapsed: {int(elapsed)}s)")
            sys.stdout.flush()
            time.sleep(0.1)
            elapsed += 0.1


def run_optimization(config: OptimizerConfig):
    import mlflow
    from dspy.evaluate import Evaluate
    from sklearn.model_selection import train_test_split

    from libraries.dspy_copro import SimpleCOPRO, save_and_log_optimized_instructions
    from libraries.ml.dspy.language_model import dspy_set_language_model

    logger = get_console_logger(f"{config.agent_name}_optimizer")

    _lm = dspy_set_language_model(config.settings, overwrite_cache_enabled=False)

    mlflow.set_experiment(config.experiment_name)
    run_name = f"{config.agent_name}_optimization_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "model_name": f"{config.agent_name}_agent",
                "optimizer": "SimpleCOPRO",
                "language_model": config.settings.language_model,
            }
        )

        mlflow.dspy.autolog(
            log_compiles=True,
            log_traces=True,
            log_evals=True,
            log_traces_from_compile=True,
            log_traces_from_eval=True,
        )

        logger.info(f"Creating {config.agent_name} dataset...")
        raw_examples = config.dataset_creator()
        examples = config.dataset_converter(raw_examples)

        logger.info(f"Created {len(examples)} examples, splitting into train/test...")
        train_set, test_set = train_test_split(
            examples,
            test_size=0.1,
            random_state=42,
        )
        if len(train_set) > 3:
            train_set = train_set[:3]

        mlflow.log_params(
            {
                "total_examples": len(examples),
                "train_examples": len(train_set),
                "test_examples": len(test_set),
            }
        )

        logger.info("Setting up evaluator...")
        evaluator = Evaluate(
            devset=test_set[:2],
            metric=config.metric_fn,
            num_threads=1,
        )

        logger.info("Evaluating baseline...")
        print_progress("Evaluating baseline")

        def evaluate_with_progress(program):
            spinner_thread = threading.Thread(
                target=_show_spinner, args=("Evaluating baseline...", 60)
            )
            spinner_thread.daemon = True
            spinner_thread.start()

            result = evaluator(program)
            sys.stdout.write("\r" + " " * 60 + "\r")
            return result

        base_score = evaluate_with_progress(config.program)
        logger.info(f"Baseline score: {base_score:.3f}")
        mlflow.log_metric("baseline_score", base_score)

        logger.info("Setting up SimpleCOPRO optimizer...")
        breadth = 2
        depth = 1

        optimizer = SimpleCOPRO(
            metric=config.metric_fn,
            breadth=breadth,
            depth=depth,
            logger=logger,
            verbose=False,
        )

        mlflow.log_params(
            {
                "copro_breadth": breadth,
                "copro_depth": depth,
            }
        )

        logger.info("Starting optimization...")
        try:
            print("\nStarting SimpleCOPRO optimization...")

            def compile_with_progress():
                print_progress("Setting up optimization")
                print_progress("Starting optimization", 0, breadth * depth)

                spinner_thread = threading.Thread(
                    target=_show_spinner, args=("Optimizing...", 120)
                )
                spinner_thread.daemon = True
                spinner_thread.start()

                result = optimizer.compile(
                    config.program,
                    trainset=train_set,
                    eval_kwargs={},
                    critical_text=config.critical_privacy_text,
                )

                print_progress(
                    "Optimization complete", breadth * depth, breadth * depth
                )
                print()

                return result

            logger.info("Using SimpleCOPRO optimization")
            optimized_program = compile_with_progress()

            logger.info("Evaluating optimized program...")
            print_progress("Evaluating optimized program")
            score = evaluate_with_progress(optimized_program)
            logger.info(
                f"Optimized score: {score:.3f} (improvement: {score - base_score:.3f})"
            )

            mlflow.log_metric("optimized_score", score)
            mlflow.log_metric("improvement", score - base_score)

            logger.info("Saving optimized program...")
            print_progress("Saving optimized program")
            json_path = Path(config.output_filename)
            sys.stdout.write("\r" + " " * 60 + "\r")

            save_result = save_and_log_optimized_instructions(
                path=json_path,
                optimized_program=optimized_program,
                original_program=config.program,
                logger=logger,
                mlflow=mlflow,
            )

            if save_result:
                logger.info("Successfully saved and logged optimized instructions")
            else:
                logger.warning(
                    "There were issues saving or logging optimized instructions"
                )

            sys.stdout.write("\r" + " " * 60 + "\r")

        except Exception as e:
            sys.stdout.write("\r" + " " * 60 + "\r")
            logger.error(f"Optimization failed: {e}")
            mlflow.log_param("optimization_error", str(e))

            print_progress("Saving fallback program")
            json_path = Path(config.output_filename)
            sys.stdout.write("\r" + " " * 60 + "\r")

            save_result = save_and_log_optimized_instructions(
                path=json_path,
                optimized_program=config.program,
                original_program=None,
                logger=logger,
                mlflow=mlflow,
            )

            if save_result:
                logger.info("Successfully saved and logged fallback instructions")
            else:
                logger.warning(
                    "There were issues with fallback save - creating emergency file"
                )

                try:
                    with open(str(json_path), "w") as f:
                        f.write(
                            '{"instructions": "'
                            + config.emergency_instruction
                            + '", "fields": []}'
                        )
                    logger.info("Created emergency minimal fallback file")
                except Exception as emergency_error:
                    logger.error(f"Emergency fallback failed: {emergency_error}")
