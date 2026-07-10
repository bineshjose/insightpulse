# =============================================================================
# InsightPulse — Makefile
# =============================================================================
# Provides convenient shortcuts for common development tasks.
# Usage: make <target>
# =============================================================================

.PHONY: help demo prod test lint format generate-data experiments clean

# Default target — show help
help: ## Show this help message
	@echo "InsightPulse — Available Commands"
	@echo "================================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# --- Environment ---

demo: ## Start the demo environment (Docker Desktop)
	@echo "🚀 Starting InsightPulse in demo mode..."
	@cp -n .env.example .env 2>/dev/null || true
	@test -f data/synthetic/panelists.csv || \
		echo "⚠  Sample data missing — run 'make generate-data' first for full demo functionality."
	docker compose -f docker-compose.yml -f docker-compose.demo.yml up --build

demo-detached: ## Start demo in background
	@cp -n .env.example .env 2>/dev/null || true
	docker compose -f docker-compose.yml -f docker-compose.demo.yml up --build -d

prod: ## Start the production environment
	docker compose -f docker-compose.yml -f docker-compose.prod.yml up --build

stop: ## Stop all containers
	docker compose down

# --- Development ---

install: ## Install dependencies for local development
	pip install -e ".[dev]"
	pre-commit install

test: ## Run test suite with coverage
	pytest tests/ -v --cov=src/insightpulse --cov-report=term-missing

test-fast: ## Run tests without coverage (faster)
	pytest tests/ -v -x

lint: ## Run linter (ruff)
	ruff check src/ tests/ dashboard/ experiments/
	mypy src/insightpulse/

format: ## Auto-format code
	ruff format src/ tests/ dashboard/ experiments/
	ruff check --fix src/ tests/ dashboard/ experiments/

# --- Data ---

generate-data: ## Generate synthetic sample data for demo mode
	python -m data.synthetic.generate_sample_data

# --- Experiments ---

experiments: ## Run all experiments (multi-LLM comparison, calibration, etc.)
	python -m experiments.multi_llm_comparison
	python -m experiments.calibration_convergence
	python -m experiments.sequential_dependency
	python -m experiments.drift_detection

experiment-llm: ## Run multi-LLM comparison experiment only
	python -m experiments.multi_llm_comparison

experiment-calibration: ## Run BDCL calibration convergence experiment
	python -m experiments.calibration_convergence

# --- Cleanup ---

clean: ## Remove build artifacts and caches
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ *.egg-info/

clean-all: clean ## Remove everything including Docker volumes
	docker compose down -v --remove-orphans
