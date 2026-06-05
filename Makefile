.DEFAULT_GOAL := help

# ── Variables ──────────────────────────────────────────────────────────────────
VENV          := .venv
SYSTEM_PYTHON := $(shell command -v python3.12 2>/dev/null || command -v python3.13 2>/dev/null || command -v python3.11 2>/dev/null || command -v python3 2>/dev/null || command -v python 2>/dev/null)
PYTHON        := $(VENV)/bin/python3
PYTEST        := $(PYTHON) -m pytest
PIP           := $(PYTHON) -m pip
BLACK         := $(PYTHON) -m black
FLAKE8        := $(PYTHON) -m flake8
SRC      := backend
TESTS    := tests
COV_MIN  := 70

.PHONY: help install hooks \
        backend frontend \
        up down build \
        test test-unit test-integration \
        format lint check \
        evals evals-full \
        version clean

# ── Help ───────────────────────────────────────────────────────────────────────
help: ## Show available targets
	@grep -E '^[a-zA-Z_%-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

# ── Setup ──────────────────────────────────────────────────────────────────────
install: ## Create .venv (if needed) and install runtime + dev dependencies
	@test -f $(PYTHON) || $(SYSTEM_PYTHON) -m venv $(VENV)
	$(PIP) install -r requirements.txt -r requirements-dev.txt

hooks: ## Install pre-commit hooks (run once after clone)
	pre-commit install --hook-type pre-commit --hook-type pre-push

# ── Local dev ──────────────────────────────────────────────────────────────────
backend: ## Start backend dev server on :8000 (requires ANTHROPIC_API_KEY)
	@test -n "$(ANTHROPIC_API_KEY)" || (echo "Error: ANTHROPIC_API_KEY is not set"; exit 1)
	cd $(SRC) && ../$(PYTHON) -m uvicorn main:app --reload --port 8000

frontend: ## Start Streamlit frontend on :8501 (requires backend running)
	cd frontend && ../$(PYTHON) -m streamlit run app.py

# ── Docker ─────────────────────────────────────────────────────────────────────
up: ## Build and start all services via Docker Compose
	@test -n "$(ANTHROPIC_API_KEY)" || (echo "Error: ANTHROPIC_API_KEY is not set"; exit 1)
	docker compose up --build

down: ## Stop and remove all Docker Compose services
	docker compose down

build: ## Build Docker images without starting containers
	docker compose build

# ── Tests ──────────────────────────────────────────────────────────────────────
test: ## Run full test suite with coverage (min $(COV_MIN)%)
	$(PYTEST) $(TESTS) -v \
		--cov=$(SRC) \
		--cov-report=term-missing \
		--cov-fail-under=$(COV_MIN)

test-unit: ## Run unit tests only
	$(PYTEST) $(TESTS)/unit -v

test-integration: ## Run integration tests only
	$(PYTEST) $(TESTS)/integration -v

# ── Code quality ───────────────────────────────────────────────────────────────
format: ## Auto-format code with Black
	$(BLACK) $(SRC)/ $(TESTS)/

lint: ## Lint with Flake8
	$(FLAKE8) $(SRC)/ $(TESTS)/

check: ## Check formatting + linting without modifying files (same as CI)
	$(BLACK) --check $(SRC)/ $(TESTS)/
	$(FLAKE8) $(SRC)/ $(TESTS)/

# ── Evals ──────────────────────────────────────────────────────────────────────
evals: ## Run keyword-match evals — fast, no API calls (backend must be running)
	$(PYTHON) evals/run_evals.py --backend http://localhost:8000 --skip-ragas

evals-full: ## Run full evals including Ragas. EVAL_LLM=huggingface uses HuggingFace instead of Anthropic
	@if [ "$(EVAL_LLM)" = "huggingface" ]; then \
		test -n "$(HUGGINGFACE_API_KEY)" || (echo "Error: HUGGINGFACE_API_KEY is not set"; exit 1); \
	else \
		test -n "$(ANTHROPIC_API_KEY)" || (echo "Error: ANTHROPIC_API_KEY is not set"; exit 1); \
	fi
	$(PYTHON) evals/run_evals.py --backend http://localhost:8000 --eval-llm $(or $(EVAL_LLM),anthropic)

# ── Version ────────────────────────────────────────────────────────────────────
version: ## Show current project version
	@$(PYTHON) -c "exec(open('backend/__version__.py').read()); print(__version__)"

# ── Clean ──────────────────────────────────────────────────────────────────────
clean: ## Remove caches, build artifacts, and coverage output
	find . -type d -name __pycache__ -not -path './.git/*' -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
	rm -f .coverage coverage.xml
