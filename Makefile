# =============================================================================
# EggAI Demo
# =============================================================================
# Core commands for running the demo. For advanced operations, use the
# individual scripts directly or see the full command reference in docs/.
# =============================================================================

.PHONY: start start-foreground stop test test-ci test-all lint lint-fix clean help \
        docker-up docker-down health benchmark-classifiers

# Default target
.DEFAULT_GOAL := help

# -----------------------------------------------------------------------------
# Core Commands
# -----------------------------------------------------------------------------

start: ## Start everything (infrastructure + agents)
	@PYTHONPATH=$(PWD) uv run scripts/start.py

start-foreground: ## Start with agent logs visible (Ctrl+C to stop)
	@PYTHONPATH=$(PWD) uv run scripts/start.py --foreground

stop: ## Stop agents (keeps Docker running)
	@PYTHONPATH=$(PWD) uv run scripts/stop.py

stop-all: ## Stop agents and Docker infrastructure
	@PYTHONPATH=$(PWD) uv run scripts/stop.py --all

health: ## Check health of all services
	@PYTHONPATH=$(PWD) uv run scripts/health_check.py

# -----------------------------------------------------------------------------
# Development
# -----------------------------------------------------------------------------

test: ## Run CI-safe tests (no external dependencies)
	@uv run pytest

test-ci: test ## Alias for test

test-all: ## Run all tests including integration
	@uv run pytest -m ""

test-coverage: ## Run tests with coverage report
	@uv run pytest --cov=agents --cov=libraries --cov-report=term --cov-report=html

lint: ## Check code quality
	@uv run ruff check agents libraries scripts

lint-fix: ## Auto-fix lint issues
	@uv run ruff check --fix agents libraries scripts

# -----------------------------------------------------------------------------
# Infrastructure
# -----------------------------------------------------------------------------

docker-up: ## Start Docker infrastructure only
	@docker compose up -d

docker-down: ## Stop Docker infrastructure
	@docker compose down

docker-reset: ## Stop and remove Docker volumes (full reset)
	@docker compose down -v

# -----------------------------------------------------------------------------
# Classifiers
# -----------------------------------------------------------------------------

benchmark-classifiers: ## Benchmark all triage classifiers
	@echo "Benchmarking classifiers (requires infrastructure)..."
	@uv run -m agents.triage.dspy_modules.evaluation.evaluate

# Training commands (for advanced users)
train-v3: ## Train few-shot classifier v3
	@uv run -m agents.triage.baseline_model.fewshot_trainer

train-v5: ## Train attention network classifier v5
	@uv run -m agents.triage.attention_net.attention_net_trainer

train-v6: ## Train OpenAI fine-tuned classifier v6
	@echo "Requires OPENAI_API_KEY environment variable"
	@uv run -m agents.triage.classifier_v6.finetune_trainer

train-v7: ## Train Gemma fine-tuned classifier v7
	@uv run -m agents.triage.classifier_v7.finetune_trainer

# -----------------------------------------------------------------------------
# Setup & Cleanup
# -----------------------------------------------------------------------------

setup: ## Install dependencies with uv
	@uv sync

clean: stop ## Stop agents and clean Python artifacts
	@rm -rf __pycache__ .pytest_cache .ruff_cache .coverage htmlcov
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true

full-reset: stop-all clean ## Full reset: stop everything, clean artifacts, remove volumes
	@docker compose down -v
	@rm -rf .venv uv.lock
	@echo "Full reset complete. Run 'make start' to begin fresh."

# -----------------------------------------------------------------------------
# Help
# -----------------------------------------------------------------------------

help: ## Show this help message
	@echo ""
	@echo "EggAI Demo - Multi-Agent Insurance Support System"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Core Commands:"
	@grep -E '^(start|start-foreground|stop|stop-all|health):.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Development:"
	@grep -E '^(test|test-ci|test-all|test-coverage|lint|lint-fix):.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Infrastructure:"
	@grep -E '^(docker-up|docker-down|docker-reset):.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Classifiers:"
	@grep -E '^(benchmark-classifiers|train-v[0-9]+):.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Setup & Cleanup:"
	@grep -E '^(setup|clean|full-reset):.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Quick Start:"
	@echo "  make start        # Start everything"
	@echo "  make stop         # Stop agents"
	@echo "  make test         # Run tests"
	@echo ""
