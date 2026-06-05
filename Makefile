.DEFAULT_GOAL := help

# ── Variables ──────────────────────────────────────────────────────────────────
PYTHON   := python
PYTEST   := $(PYTHON) -m pytest
PIP      := $(PYTHON) -m pip
BLACK    := $(PYTHON) -m black
FLAKE8   := $(PYTHON) -m flake8
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
install: ## Install runtime + dev dependencies
	$(PIP) install -r requirements.txt -r requirements-dev.txt

hooks: ## Install pre-commit hooks (run once after clone)
	pre-commit install --hook-type pre-commit --hook-type pre-push

# ── Local dev ──────────────────────────────────────────────────────────────────
backend: ## Start backend dev server on :8000 (requires ANTHROPIC_API_KEY)
	@test -n "$(ANTHROPIC_API_KEY)" || (echo "Error: ANTHROPIC_API_KEY is not set"; exit 1)
	cd $(SRC) && uvicorn main:app --reload --port 8000

frontend: ## Start Streamlit frontend on :8501 (requires backend running)
	cd frontend && streamlit run app.py

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

evals-full: ## Run full evals including Ragas (requires ANTHROPIC_API_KEY + backend)
	@test -n "$(ANTHROPIC_API_KEY)" || (echo "Error: ANTHROPIC_API_KEY is not set"; exit 1)
	$(PYTHON) evals/run_evals.py --backend http://localhost:8000

# ── Version ────────────────────────────────────────────────────────────────────
version: ## Show current project version
	@$(PYTHON) -c "exec(open('backend/__version__.py').read()); print(__version__)"

# ── Clean ──────────────────────────────────────────────────────────────────────
clean: ## Remove caches, build artifacts, and coverage output
	find . -type d -name __pycache__ -not -path './.git/*' -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
	rm -f .coverage coverage.xml
