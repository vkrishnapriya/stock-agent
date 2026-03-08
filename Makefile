.PHONY: help install install-dev lint format typecheck test test-integration \
        test-cov run setup-kite docker-up docker-down clean

PYTHON   := python
PIP      := $(PYTHON) -m pip
PYTEST   := $(PYTHON) -m pytest
RUFF     := $(PYTHON) -m ruff
MYPY     := $(PYTHON) -m mypy

help:          ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:       ## Install production dependencies
	$(PIP) install -e .

install-dev:   ## Install all dependencies including dev extras
	$(PIP) install -e ".[dev]"
	pre-commit install

lint:          ## Lint with ruff
	$(RUFF) check src tests scripts

format:        ## Auto-format with ruff
	$(RUFF) format src tests scripts
	$(RUFF) check --fix src tests scripts

typecheck:     ## Run mypy type checks
	$(MYPY) src

test:          ## Run unit tests (no external services)
	$(PYTEST) tests/unit -v

test-integration: ## Run integration tests (needs Redis + Postgres)
	$(PYTEST) tests/integration -v

test-cov:      ## Run all tests with coverage report
	$(PYTEST) tests/unit --cov=src --cov-report=html --cov-report=term-missing

run:           ## Run the trading agent (paper mode)
	$(PYTHON) -m src.main

setup-kite:    ## Refresh Kite Connect access token
	$(PYTHON) scripts/setup_kite.py

docker-up:     ## Start Redis + Postgres via Docker Compose
	docker compose up -d

docker-down:   ## Stop Docker services
	docker compose down

clean:         ## Remove build artifacts and caches
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage dist build
