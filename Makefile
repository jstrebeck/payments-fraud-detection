# Task runner. `make help` lists targets.

SHELL := bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help

GIT_SHA  := $(shell git rev-parse --short HEAD 2>/dev/null || echo dev)
REGISTRY ?= 192.168.2.203:5000/fraud
COMPOSE  := docker compose
SEED     ?= 42
SIM_ARGS ?= --rps 20 --duration 60s
API_URL  ?= http://localhost:8000
IMAGES   := payments-api simulator
# Serving runtime image (ADR-0014): tagged by its stack, not the git SHA.
SERVING_TAG ?= mlserver-1.7.1-mlflow-3.16.1
DATA     := data/transactions.parquet

# MLflow for host tools (train, promote). Defaults to the homelab server
# (ADR-0012); override in .env or on the command line. Other settings are read
# from .env by the tools themselves.
-include .env
MLFLOW_TRACKING_URI ?= http://192.168.2.202
export MLFLOW_TRACKING_URI
export MLFLOW_DISABLE_AGENT_HINT := 1

.PHONY: help install lint fmt typecheck test check dev dev-mlflow down clean-dev logs psql \
        generate train promote simulate api build push smoke smoke-cluster serving-image manifests \
        release train-cluster

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

## --- Python -----------------------------------------------------------------

install: ## Install all workspace packages and the pre-commit hooks
	uv sync --all-packages
	uv run pre-commit install

lint: ## ruff lint + format check, then mypy
	uv run ruff check .
	uv run ruff format --check .
	$(MAKE) --no-print-directory typecheck

fmt: ## Auto-format and apply safe lint fixes
	uv run ruff format .
	uv run ruff check --fix .

typecheck: ## mypy --strict, one run per workspace member
	uv run mypy ml
	uv run mypy services/payments-api/payments_api services/payments-api/tests
	uv run mypy services/simulator/simulator services/simulator/tests

test: ## pytest with coverage
	uv run pytest --cov --cov-report=term

check: lint test ## Everything CI runs

## --- Local stack ------------------------------------------------------------

dev: ## Start postgres and the API (docker compose); MLflow is the homelab one
	$(COMPOSE) up -d --build --wait postgres payments-api
	@echo "API     $(API_URL)/docs"
	@echo "MLflow  $${MLFLOW_TRACKING_URI:-http://192.168.2.202} (homelab; make dev-mlflow for a local one)"

dev-mlflow: ## Also start a local MLflow + S3 (for working without the homelab)
	API_MLFLOW_TRACKING_URI=http://mlflow:5000 \
		$(COMPOSE) --profile local-mlflow up -d --build --wait postgres s3 mlflow payments-api
	@echo "MLflow  http://localhost:5000  (export MLFLOW_TRACKING_URI=http://localhost:5000)"

down: ## Stop the local stack (keeps data volumes)
	$(COMPOSE) --profile sim --profile local-mlflow down

clean-dev: ## Stop the local stack and delete its volumes
	$(COMPOSE) --profile sim --profile local-mlflow down -v

logs: ## Follow API logs
	$(COMPOSE) logs -f payments-api

psql: ## psql into the payments database
	$(COMPOSE) exec postgres psql -U fraud payments

generate: ## Write data/transactions.parquet (SEED=42)
	uv run python -m ml.data generate --seed $(SEED) --out $(DATA)

$(DATA):
	$(MAKE) --no-print-directory generate

## --- Model ------------------------------------------------------------------

train: $(DATA) ## Train, log and register a version in MLflow (MLFLOW_TRACKING_URI)
	uv run python -m ml.training

promote: ## Gate the latest version; move `champion` on a win (PROMOTE_ARGS="--dry-run")
	uv run python scripts/promote.py $(PROMOTE_ARGS)

simulate: ## Replay synthetic traffic against the compose API (SIM_ARGS="--rps 20 --duration 60s")
	$(COMPOSE) --profile sim run --rm --build simulator run --seed $(SEED) $(SIM_ARGS)

api: ## Run the API on the host with hot reload (after `make dev`; stop the compose API first)
	DATABASE_URL=postgresql+asyncpg://fraud:fraud@localhost:5432/payments \
		uv run uvicorn payments_api.main:create_app --factory --reload --port 8000

smoke: ## /healthz, /readyz and one POST /payments against API_URL
	scripts/smoke.sh $(API_URL)

smoke-cluster: ## Smoke test the in-cluster API via port-forward; requires a KServe score
	scripts/smoke-cluster.sh

## --- Kubernetes -------------------------------------------------------------

manifests: ## Render deploy/overlays/homelab (what Argo CD will sync); no cluster needed
	kubectl kustomize deploy/overlays/homelab > /dev/null
	@echo "deploy/overlays/homelab renders"

## --- Images -----------------------------------------------------------------

build: ## Build images tagged with the git SHA
	for img in $(IMAGES); do \
		docker build -f services/$$img/Dockerfile -t $(REGISTRY)/$$img:$(GIT_SHA) . ; \
	done

push: build ## Push images to the homelab registry
	for img in $(IMAGES); do docker push $(REGISTRY)/$$img:$(GIT_SHA); done

## --- Delivery (docs/ci-cd.md) ----------------------------------------------
# CI runs these same scripts on the self-hosted runner. While the runner is
# offline, run them by hand; Argo CD deploys whatever the overlay pins.

release: ## Build+push images for HEAD, pin them in the homelab overlay, commit and push
	scripts/release.sh
	git add deploy/overlays/homelab/kustomization.yaml
	git commit -m "chore(deploy): images $$(git rev-parse --short HEAD) [skip ci]"
	git push

train-cluster: ## Run a training Job in the cluster (SEED=, CUSTOMERS=, DAYS=); registers, does not promote
	scripts/train-cluster.sh

serving-image: ## Build and push the KServe serving runtime (ml/serving, ADR-0014)
	docker build -f ml/serving/Dockerfile -t $(REGISTRY)/serving:$(SERVING_TAG) ml/serving
	docker push $(REGISTRY)/serving:$(SERVING_TAG)
