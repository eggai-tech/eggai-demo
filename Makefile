# =============================================================================
# EggAI Demo
# =============================================================================
# Core commands for running the demo. For advanced operations, use the
# individual scripts directly or see the full command reference in docs/.
# =============================================================================

.PHONY: start start-foreground stop stop-all test test-ci test-all test-coverage \
        lint lint-fix clean full-reset help setup \
        docker-up docker-down docker-reset health benchmark-classifiers \
        security-scan sast-scan deps-export type-check type-check-all \
        train-v3 train-v5 train-v6 train-v7 \
        kind-up kind-down kind-recreate kind-repos kind-infra \
        kind-llm kind-app kind-build kind-redeploy kind-deploy kind-dashboards \
        kind-status kind-gateway kind-gateway-api kind-urls kind-clean kind-destroy

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

# Modules the type check gates in CI. Type checking is being adopted
# incrementally: run `make type-check-all` to see the remaining backlog, clean
# up a package, then add it here so it cannot regress.
# agents/triage is not yet gated - see its classifiers/, data_sets/, and
# evaluation/ backlog via `make type-check-all`.
TYPED_MODULES := libraries scripts \
	agents/billing agents/claims agents/escalation agents/frontend agents/audit agents/policies

# --warnings makes pyright exit non-zero on warnings too, so the gated modules
# stay at zero diagnostics rather than slowly accruing ignored warnings.
type-check: ## Type-check the gated modules with pyright (fails on any diagnostic)
	@uv run pyright --warnings $(TYPED_MODULES)

type-check-all: ## Type-check everything, including modules not yet gated
	@uv run pyright agents libraries scripts

# -----------------------------------------------------------------------------
# Dependencies & Security
# -----------------------------------------------------------------------------

security-scan: ## Scan dependencies for known vulnerabilities (fails on HIGH/CRITICAL)
	@uv run --no-project scripts/security_scan.py

sast-scan: ## Scan source for insecure code patterns with opengrep (fails on HIGH/CRITICAL)
	@uv run --no-project scripts/sast_scan.py

deps-export: ## Regenerate requirements.txt / dev-requirements.txt from uv.lock
	@uv export --no-dev --no-emit-project --format requirements-txt -o requirements.txt
	@uv export --only-dev --no-emit-project --format requirements-txt -o dev-requirements.txt
	@echo "requirements.txt and dev-requirements.txt regenerated from uv.lock"

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
# Local Kubernetes (kind)
# -----------------------------------------------------------------------------
# Runs the stack in a real cluster instead of docker compose, so probes,
# resource limits, consumer-group rebalancing across replicas and rolling
# updates are all exercised for real. Manifests live in kind/.
#
# Every component is a toggle. Setting one to false UNINSTALLS it rather than
# skipping it, so the cluster always matches the flags:
#   make kind-deploy KIND_TEMPORAL=false KIND_TEMPO=false

KIND_DIR           := kind
KIND_CLUSTER       ?= eggai
KIND_NODE_IMAGE    ?= kindest/node:v1.31.2
KIND_APP_NS        ?= eggai-demo
KIND_OBS_NS        ?= observability
KIND_REGISTRY_PORT ?= 5001
KIND_REGISTRY      := localhost:$(KIND_REGISTRY_PORT)
# Always target the kind cluster explicitly, never the current kubeconfig context
KUBECTL            := kubectl --context kind-$(KIND_CLUSTER)
HELM               := helm --kube-context kind-$(KIND_CLUSTER)
# Override when auto-detection finds the wrong host (e.g. macOS/Colima, where
# host.docker.internal resolves to the Lima VM rather than the Mac).
KIND_LLM_HOST_IP ?=

KIND_TRAEFIK_VER     ?= 41.3.0
KIND_KUBEPROM_VER    ?= 68.3.0
KIND_REDPANDA_VER    ?= 26.2.2
KIND_GATEWAY_API_VER ?= v1.5.1
KIND_TEMPO_VER       ?= 1.18.2
KIND_OTEL_VER        ?= 0.108.0

KIND_GIT_SHA    := $(shell git rev-parse --short HEAD)
KIND_IMAGE_REPO ?= $(KIND_REGISTRY)/eggai-demo
KIND_IMAGE_TAG  ?= dev-$(KIND_GIT_SHA)

# Component toggles -- lean by default; opt into observability when needed.
KIND_TRAEFIK    ?= true
KIND_REDPANDA   ?= true
KIND_APP        ?= true
KIND_OTEL       ?= true
KIND_PROMETHEUS ?= true
KIND_TEMPO      ?= true
KIND_TEMPORAL   ?= false

# $(call kind_helm,release,toggle,chart,namespace,values-file,extra-flags)
define kind_helm
@if [ "$(2)" = "true" ]; then \
	echo "==> $(1)"; \
	$(HELM) upgrade --install $(1) $(3) -n $(4) --create-namespace \
		-f $(KIND_DIR)/$(5) $(6); \
else \
	echo "--- $(1) (disabled)"; \
	$(HELM) uninstall $(1) -n $(4) >/dev/null 2>&1 || true; \
fi
endef

kind-up: ## Create the kind cluster and local image registry
	@docker inspect kind-registry >/dev/null 2>&1 || \
		docker run -d --restart=always -p 127.0.0.1:$(KIND_REGISTRY_PORT):5000 \
			--name kind-registry registry:2
	@kind get clusters 2>/dev/null | grep -qx $(KIND_CLUSTER) || \
		kind create cluster --name $(KIND_CLUSTER) --image $(KIND_NODE_IMAGE) \
			--config $(KIND_DIR)/kind.yaml
	@docker network inspect kind | grep -q kind-registry || \
		docker network connect kind kind-registry
	@for n in $$(kind get nodes --name $(KIND_CLUSTER)); do \
		docker exec $$n bash -c 'mkdir -p /etc/containerd/certs.d/localhost:$(KIND_REGISTRY_PORT) && \
			printf "[host.\"http://kind-registry:5000\"]\n" > /etc/containerd/certs.d/localhost:$(KIND_REGISTRY_PORT)/hosts.toml'; \
	done
	@echo "Cluster $(KIND_CLUSTER) ready. Registry on $(KIND_REGISTRY)."

kind-down: ## Delete the kind cluster (registry survives, keeping its layer cache)
	@kind delete cluster --name $(KIND_CLUSTER)

kind-recreate: kind-down kind-up kind-deploy ## Rebuild the cluster from scratch and redeploy

kind-repos: ## Add/update the Helm repos the local stack pulls from
	@helm repo add traefik https://traefik.github.io/charts >/dev/null 2>&1 || true
	@helm repo add prometheus-community https://prometheus-community.github.io/helm-charts >/dev/null 2>&1 || true
	@helm repo add open-telemetry https://open-telemetry.github.io/opentelemetry-helm-charts >/dev/null 2>&1 || true
	@helm repo add grafana https://grafana.github.io/helm-charts >/dev/null 2>&1 || true
	@helm repo add redpanda https://charts.redpanda.com >/dev/null 2>&1 || true
	@helm repo update >/dev/null

# --wait on kube-prom is load-bearing: the ServiceMonitor CRD has to exist
# before any later chart renders one, or the release fails on an unknown kind.
kind-infra: kind-repos ## Deploy enabled infrastructure components only
	$(call kind_helm,kube-prom,$(KIND_PROMETHEUS),prometheus-community/kube-prometheus-stack,$(KIND_OBS_NS),kube-prom-kind.yaml,--version $(KIND_KUBEPROM_VER) --wait --timeout 10m)
	@[ "$(KIND_PROMETHEUS)" != "true" ] || $(MAKE) --no-print-directory kind-dashboards
	@$(MAKE) --no-print-directory kind-gateway-api
	$(call kind_helm,traefik,$(KIND_TRAEFIK),traefik/traefik,traefik,traefik-kind.yaml,--version $(KIND_TRAEFIK_VER))
	@if [ "$(KIND_TRAEFIK)" = "true" ]; then \
		echo "==> gateway"; \
		$(KUBECTL) apply -f $(KIND_DIR)/gateway-kind.yaml; \
	else \
		echo "--- gateway (disabled)"; \
		$(KUBECTL) delete -f $(KIND_DIR)/gateway-kind.yaml --ignore-not-found >/dev/null 2>&1 || true; \
	fi
	$(call kind_helm,tempo,$(KIND_TEMPO),grafana/tempo,$(KIND_OBS_NS),tempo-kind.yaml,--version $(KIND_TEMPO_VER) --set serviceMonitor.enabled=$(KIND_PROMETHEUS))
	$(call kind_helm,otel-collector,$(KIND_OTEL),open-telemetry/opentelemetry-collector,$(KIND_OBS_NS),otel-collector-kind.yaml,--version $(KIND_OTEL_VER) --set serviceMonitor.enabled=$(KIND_PROMETHEUS))
	$(call kind_helm,redpanda,$(KIND_REDPANDA),redpanda/redpanda,$(KIND_APP_NS),redpanda-kind.yaml,--version $(KIND_REDPANDA_VER) --set monitoring.enabled=$(KIND_PROMETHEUS))
	@$(MAKE) --no-print-directory kind-llm
	@if [ "$(KIND_TEMPORAL)" = "true" ]; then \
		echo "==> temporal"; \
		$(KUBECTL) create ns $(KIND_APP_NS) --dry-run=client -o yaml | $(KUBECTL) apply -f - >/dev/null; \
		$(KUBECTL) apply -n $(KIND_APP_NS) -f $(KIND_DIR)/temporal-kind.yaml; \
	else \
		echo "--- temporal (disabled)"; \
		$(KUBECTL) delete -n $(KIND_APP_NS) -f $(KIND_DIR)/temporal-kind.yaml --ignore-not-found >/dev/null 2>&1 || true; \
	fi

KIND_APP_FLAGS = --set image.repository=$(KIND_IMAGE_REPO) \
                 --set image.tag=$(KIND_IMAGE_TAG) \
                 --set image.pullPolicy=Always \
                 --set monitoring.enabled=$(KIND_PROMETHEUS) \
                 --wait --timeout 5m \
                 $(if $(filter true,$(KIND_OTEL)),--set globalEnv.OTEL_ENDPOINT=http://otel-collector.$(KIND_OBS_NS).svc.cluster.local:4318)

kind-app: kind-build kind-gateway ## Build, push and deploy the app -- the inner loop
	$(call kind_helm,eggai,$(KIND_APP),./helm,$(KIND_APP_NS),values-kind.yaml,$(KIND_APP_FLAGS))
	@$(MAKE) --no-print-directory kind-urls

kind-redeploy: kind-gateway ## Redeploy the app chart without rebuilding
	$(call kind_helm,eggai,$(KIND_APP),./helm,$(KIND_APP_NS),values-kind.yaml,$(KIND_APP_FLAGS))
	@$(MAKE) --no-print-directory kind-urls

kind-gateway: ## Deploy the HTTPRoutes (Gateway comes from the Traefik chart)
	@if [ "$(KIND_TRAEFIK)" = "true" ]; then \
		echo "==> httproute"; \
		$(KUBECTL) apply -n $(KIND_APP_NS) -f $(KIND_DIR)/httproute-kind.yaml; \
		[ "$(KIND_PROMETHEUS)" != "true" ] || \
			$(KUBECTL) apply -n $(KIND_OBS_NS) -f $(KIND_DIR)/httproute-obs-kind.yaml; \
	else \
		$(KUBECTL) delete -n $(KIND_APP_NS) -f $(KIND_DIR)/httproute-kind.yaml --ignore-not-found >/dev/null 2>&1 || true; \
		$(KUBECTL) delete -n $(KIND_OBS_NS) -f $(KIND_DIR)/httproute-obs-kind.yaml --ignore-not-found >/dev/null 2>&1 || true; \
	fi

kind-llm: ## Point the cluster at LM Studio on the host
	@$(KUBECTL) create ns $(KIND_APP_NS) --dry-run=client -o yaml | $(KUBECTL) apply -f - >/dev/null
	@IP="$(KIND_LLM_HOST_IP)"; \
	if [ -z "$$IP" ]; then \
		IP=$$(docker exec $(KIND_CLUSTER)-control-plane sh -c \
			"getent hosts host.docker.internal 2>/dev/null | awk '{print \$$1}' | head -1"); \
	fi; \
	if [ -z "$$IP" ]; then \
		IP=$$(docker exec $(KIND_CLUSTER)-control-plane sh -c "ip route | awk '/default/{print \$$3}'"); \
	fi; \
	echo "==> llm -> host LM Studio at $$IP:1234"; \
	sed "s|__HOST_IP__|$$IP|" $(KIND_DIR)/llm-host-kind.yaml | $(KUBECTL) apply -n $(KIND_APP_NS) -f -

kind-gateway-api: ## Install Gateway API CRDs (chart will stop shipping them)
	@echo "==> gateway-api $(KIND_GATEWAY_API_VER)"
	@$(KUBECTL) apply -f https://github.com/kubernetes-sigs/gateway-api/releases/download/$(KIND_GATEWAY_API_VER)/standard-install.yaml >/dev/null

kind-deploy: kind-infra kind-app ## Deploy the whole enabled stack

kind-build: ## Build the image and push it to the local registry
	@echo "==> building $(KIND_IMAGE_REPO):$(KIND_IMAGE_TAG)"
	@docker build -t $(KIND_IMAGE_REPO):$(KIND_IMAGE_TAG) .
	@docker push $(KIND_IMAGE_REPO):$(KIND_IMAGE_TAG)

kind-dashboards: ## Load the repo's Grafana dashboard into the cluster
	@$(KUBECTL) create configmap grafana-dash-eggai -n $(KIND_OBS_NS) \
		--from-file=dockerConfig/grafana-dashboard.json --dry-run=client -o yaml \
		| $(KUBECTL) label --local -f - grafana_dashboard=1 -o yaml \
		| $(KUBECTL) apply -f -

kind-status: ## Show pods, restart counts and OOMKills across all namespaces
	@$(KUBECTL) get pods -A -o custom-columns=\
NS:.metadata.namespace,NAME:.metadata.name,READY:.status.containerStatuses[0].ready,\
RESTARTS:.status.containerStatuses[0].restartCount,\
LAST:.status.containerStatuses[0].lastState.terminated.reason

kind-urls: ## Print the local ingress hostnames for enabled components
	@echo "  app        http://chat.eggai.localhost"
	@echo "  traefik    http://traefik.eggai.localhost"
	@[ "$(KIND_REDPANDA)"   = "true" ] && echo "  redpanda   http://redpanda.eggai.localhost" || true
	@[ "$(KIND_PROMETHEUS)" = "true" ] && echo "  grafana    http://grafana.eggai.localhost   (admin/admin)" || true
	@[ "$(KIND_TEMPORAL)"   = "true" ] && echo "  temporal   http://temporal.eggai.localhost" || true

kind-clean: ## Uninstall everything but keep the cluster
	@$(MAKE) kind-infra kind-redeploy kind-gateway KIND_TRAEFIK=false \
		KIND_PROMETHEUS=false KIND_TEMPO=false KIND_OTEL=false \
		KIND_REDPANDA=false KIND_TEMPORAL=false KIND_APP=false

kind-destroy: ## Delete the cluster, registry, its volume, and local build images
	@kind delete cluster --name $(KIND_CLUSTER) 2>/dev/null || true
	@docker rm -fv kind-registry >/dev/null 2>&1 || true
	@docker image ls --format '{{.Repository}}:{{.Tag}}' \
		| grep '^$(KIND_REGISTRY)/eggai-demo:' \
		| xargs -r docker rmi -f >/dev/null 2>&1 || true
	@echo "Cluster, registry, registry volume and local eggai-demo images removed."

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
	@grep -E '^(test|test-ci|test-all|test-coverage|lint|lint-fix|type-check|type-check-all):.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Dependencies & Security:"
	@grep -E '^(security-scan|sast-scan|deps-export):.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
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
	@echo "Local Kubernetes (kind):"
	@grep -E '^kind-[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Quick Start:"
	@echo "  make start        # Start everything"
	@echo "  make stop         # Stop agents"
	@echo "  make test         # Run tests"
	@echo ""
	@echo "  make kind-up      # Create the local cluster"
	@echo "  make kind-deploy  # Build, push and deploy the stack"
	@echo "  make kind-urls    # Where to reach it"
	@echo ""
	@echo "kind toggles (true/false):"
	@echo "  KIND_TRAEFIK KIND_REDPANDA KIND_OTEL KIND_PROMETHEUS KIND_TEMPO KIND_APP  (default true)"
	@echo "  KIND_TEMPORAL                                                             (default false)"
	@echo "  KIND_LLM_HOST_IP=<ip>   LM Studio host override (macOS/Colima)"
	@echo ""
	@echo "  make kind-deploy KIND_PROMETHEUS=false KIND_TEMPO=false"
	@echo ""

