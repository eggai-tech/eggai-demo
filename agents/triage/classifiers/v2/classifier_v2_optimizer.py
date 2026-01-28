import sys
import time
from itertools import product
from pathlib import Path

import mlflow
from dspy.evaluate import Evaluate
from dspy.teleprompt import BootstrapFewShotWithRandomSearch
from sklearn.model_selection import train_test_split

from agents.triage.classifiers.v2.classifier_v2 import classifier_v2_program
from agents.triage.data_sets.loader import (
    as_dspy_examples,
    load_dataset_triage_training,
)
from libraries.ml.dspy.language_model import dspy_set_language_model
from libraries.observability.logger import get_console_logger

logger = get_console_logger("triage_optimizer_v2")


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



def macro_f1(example, pred, trace=None):
    return float(example.target_agent.lower() == pred.target_agent.lower())


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    from agents.triage.config import Settings

    settings = Settings()

    mlflow.set_experiment("triage_classifier_optimization")
    run_name = f"classifier_v2_bootstrap_optimization_{settings.bootstrap_dataset_size}examples"

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "model_name": "classifier_v2",
                "optimizer": "BootstrapFewShotWithRandomSearch",
                "language_model": settings.language_model,
                "dataset_size": settings.bootstrap_dataset_size,
                "speed_optimized": True,
                "num_candidate_programs": 4,
                "max_labeled_demos": 8,
                "max_bootstrapped_demos": 4,
                "max_rounds": 2,
            }
        )

        mlflow.dspy.autolog(
            log_compiles=True,
            log_traces=True,
            log_evals=True,
            log_traces_from_compile=True,
            log_traces_from_eval=True,
        )
        lm = dspy_set_language_model(settings)

        logger.info("Loading and preparing dataset...")
        print_progress("Loading dataset")

        all_examples = as_dspy_examples(load_dataset_triage_training())

        train_size = min(settings.bootstrap_dataset_size, len(all_examples))
        sys.stdout.write("\r" + " " * 60 + "\r")  # Clear the spinner
        logger.info(
            f"✅ Using {train_size} examples for optimization (from total {len(all_examples)})"
        )

        print_progress("Creating train/test split")

        train_set, dev_set = train_test_split(
            all_examples[:train_size],
            test_size=0.20,
            stratify=[ex.target_agent for ex in all_examples[:train_size]],
            random_state=42,
        )

        sys.stdout.write("\r" + " " * 60 + "\r")  # Clear the spinner
        logger.info(
            f"✅ Created stratified split: {len(train_set)} train, {len(dev_set)} dev examples"
        )

        logger.info("Setting up evaluation...")
        print_progress("Creating evaluator")
        evaluator = Evaluate(devset=dev_set, metric=macro_f1, num_threads=8)
        sys.stdout.write("\r" + " " * 60 + "\r")  # Clear the spinner

        def evaluate_with_progress(program, message="Evaluating"):
            def show_spinner():
                elapsed = 0
                while elapsed < 60:
                    chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
                    for char in chars:
                        sys.stdout.write(
                            f"\r{char} {message}... (elapsed: {int(elapsed)}s)"
                        )
                        sys.stdout.flush()
                        time.sleep(0.1)
                        elapsed += 0.1

            import threading

            spinner_thread = threading.Thread(target=show_spinner)
            spinner_thread.daemon = True
            spinner_thread.start()

            result = evaluator(program)

            sys.stdout.write("\r" + " " * 60 + "\r")
            return result

        SEARCH_SPACE = {
            "max_labeled_demos": [8],
            "max_bootstrapped_demos": [4],
            "max_rounds": [2],
        }

        logger.info(f"Hyperparameter grid search space: {SEARCH_SPACE}")
        logger.info(
            f"Total configurations to try: {len(list(product(*SEARCH_SPACE.values())))}"
        )

        best_program, best_dev = None, -1

        total_configs = len(list(product(*SEARCH_SPACE.values())))
        current_config = 0

        logger.info("Evaluating baseline model before optimization...")
        base_score = evaluate_with_progress(
            classifier_v2_program, "Evaluating baseline model"
        )
        logger.info(f"✅ Baseline accuracy: {base_score:.3f}")
        mlflow.log_metric("baseline_accuracy", base_score)

        for values in product(*SEARCH_SPACE.values()):
            current_config += 1
            hp = dict(zip(SEARCH_SPACE.keys(), values, strict=False))

            logger.info(f"\n{'=' * 80}")
            logger.info(f"Testing configuration {current_config}/{total_configs}:")
            logger.info(f"Parameters: {hp}")
            logger.info(f"{'=' * 80}")

            print_progress("Grid search", current_config, total_configs)

            tele = BootstrapFewShotWithRandomSearch(
                metric=macro_f1,
                num_candidate_programs=4,
                max_errors=15,
                **hp,
            )

            logger.info(
                f"Starting compilation with {hp['max_labeled_demos']} demos, "
                f"{hp['max_bootstrapped_demos']} bootstrapped demos, and {hp['max_rounds']} rounds..."
            )

            print_progress(f"Bootstrapping examples for config {current_config}")

            try:
                def show_spinner(config_num=current_config):
                    elapsed = 0
                    while elapsed < 300:
                        chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
                        for char in chars:
                            sys.stdout.write(
                                f"\r{char} Bootstrapping examples (config {config_num})... (elapsed: {int(elapsed)}s)"
                            )
                            sys.stdout.flush()
                            time.sleep(0.1)
                            elapsed += 0.1

                import threading

                spinner_thread = threading.Thread(target=show_spinner)
                spinner_thread.daemon = True
                spinner_thread.start()

                prog = tele.compile(
                    classifier_v2_program, trainset=train_set, valset=dev_set
                )
                sys.stdout.write("\r" + " " * 60 + "\r")
                logger.info(f"✅ Successfully compiled configuration {current_config}")

                mlflow.log_params(
                    {
                        f"config_{current_config}_labeled_demos": hp[
                            "max_labeled_demos"
                        ],
                        f"config_{current_config}_bootstrapped_demos": hp[
                            "max_bootstrapped_demos"
                        ],
                        f"config_{current_config}_rounds": hp["max_rounds"],
                    }
                )

                logger.info(f"Evaluating configuration {current_config}...")
                score = evaluate_with_progress(
                    prog, f"Evaluating config {current_config}"
                )

                mlflow.log_metric(f"config_{current_config}_score", score)
                logger.info(f"✅ Configuration {current_config} score: {score:.3f}")

                if score > best_dev:
                    best_program, best_dev, best_hp = prog, score, hp
                    logger.info(f"\n{'=' * 80}")
                    logger.info("✨ NEW BEST PROGRAM FOUND! ✨")
                    logger.info(f"Best score so far: {best_dev:.3f}")
                    logger.info(f"Best hyper-parameters so far: {best_hp}")
                    logger.info(f"{'=' * 80}\n")

                    mlflow.log_params(
                        {
                            "best_labeled_demos": best_hp["max_labeled_demos"],
                            "best_bootstrapped_demos": best_hp[
                                "max_bootstrapped_demos"
                            ],
                            "best_rounds": best_hp["max_rounds"],
                        }
                    )
                    mlflow.log_metric("best_score", best_dev)

                    print_progress("Best score improvement", int(100 * best_dev), 100)
                    time.sleep(0.5)
                    sys.stdout.write("\r" + " " * 60 + "\r")
            except Exception as e:
                sys.stdout.write("\r" + " " * 60 + "\r")
                logger.error(
                    f"❌ Error during compilation of config {current_config}: {e}"
                )
                mlflow.log_param(f"config_{current_config}_error", str(e))

        json_path = Path(__file__).resolve().parent / "optimizations_v2.json"

        if best_program is not None:
            logger.info("Saving best program from optimization...")
            print_progress("Saving optimized program")

            try:
                best_program.save(str(json_path))
                sys.stdout.write("\r" + " " * 60 + "\r")
                logger.info(
                    f"✅ Saved best program ({best_dev:.3f} dev accuracy) → {json_path}"
                )

                print_progress("Logging artifacts to MLflow")
                mlflow.log_artifact(str(json_path))
                sys.stdout.write("\r" + " " * 60 + "\r")
                logger.info("✅ Artifacts logged successfully")

                if base_score > 0:
                    improvement = best_dev - base_score
                    logger.info(
                        f"📈 Total improvement: {improvement:.3f} (from {base_score:.3f} to {best_dev:.3f})"
                    )
                    mlflow.log_metric("improvement", improvement)
            except Exception as e:
                sys.stdout.write("\r" + " " * 60 + "\r")
                logger.error(f"❌ Error saving optimized program: {e}")

                print_progress("Saving fallback program")
                try:
                    classifier_v2_program.save(str(json_path))
                    sys.stdout.write("\r" + " " * 60 + "\r")
                    logger.info(f"✅ Saved original program instead → {json_path}")

                    print_progress("Logging fallback artifacts")
                    mlflow.log_artifact(str(json_path))
                    sys.stdout.write("\r" + " " * 60 + "\r")
                    logger.info("✅ Fallback artifacts logged successfully")
                except Exception as e2:
                    sys.stdout.write("\r" + " " * 60 + "\r")
                    logger.error(f"❌ Error saving original program: {e2}")
        else:
            logger.warning(
                "⚠️ No successful optimizations were performed. No program to save."
            )
