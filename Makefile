# Developer entry points.
#
# Every target here is what CI runs, so "green locally" and "green in CI" mean the same
# thing. `make help` lists them.
#
# The quick-start path from a clean checkout is:
#     make install
#     make check      # format, lint, types, tests
#     make smoke      # a real end-to-end search

.DEFAULT_GOAL := help
SHELL := /usr/bin/env bash

PYTHON ?= python3
PIP    ?= $(PYTHON) -m pip
CONFIG ?= configs/smoke_test.yaml
SMOKE_OUTPUT ?= artifacts/smoke

# Source layout, used by the linting and typing targets.
SOURCES := src/nas_engine tests examples scripts

.PHONY: help
help: ## Show this help
	@grep -hE '^[a-zA-Z0-9_-]+:.*?## ' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[1m%-18s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------- setup --
.PHONY: install
install: ## Install the package and its development dependencies (editable)
	$(PIP) install -e ".[dev]"

.PHONY: install-all
install-all: ## Install with every optional extra, including torchvision for CIFAR-10
	$(PIP) install -e ".[dev,cifar]"

.PHONY: hooks
hooks: ## Install the pre-commit hooks
	$(PYTHON) -m pre_commit install

# ----------------------------------------------------------------------- code quality --
.PHONY: format
format: ## Format the code with Ruff
	$(PYTHON) -m ruff format $(SOURCES)
	$(PYTHON) -m ruff check --fix $(SOURCES)

.PHONY: format-check
format-check: ## Verify formatting without changing anything (what CI runs)
	$(PYTHON) -m ruff format --check $(SOURCES)

.PHONY: lint
lint: ## Run the Ruff linter
	$(PYTHON) -m ruff check $(SOURCES)

.PHONY: typecheck
typecheck: ## Run mypy in strict mode
	$(PYTHON) -m mypy

.PHONY: check
check: format-check lint typecheck test ## Everything CI checks, in CI's order

# ------------------------------------------------------------------------------ tests --
.PHONY: test
test: ## Run the default test suite (excludes tests marked `slow`)
	$(PYTHON) -m pytest

.PHONY: test-unit
test-unit: ## Run unit tests only
	$(PYTHON) -m pytest tests/unit

.PHONY: test-property
test-property: ## Run property-based tests only
	$(PYTHON) -m pytest tests/property

.PHONY: test-integration
test-integration: ## Run integration tests only
	$(PYTHON) -m pytest tests/integration

.PHONY: test-e2e
test-e2e: ## Run end-to-end searches, including the slow ones
	$(PYTHON) -m pytest tests/end_to_end -m ""

.PHONY: test-recovery
test-recovery: ## Run failure and recovery tests only
	$(PYTHON) -m pytest tests/failure_recovery

.PHONY: test-regression
test-regression: ## Run golden-fixture and determinism tests only
	$(PYTHON) -m pytest tests/regression

.PHONY: test-performance
test-performance: ## Run the coarse performance guards
	$(PYTHON) -m pytest tests/performance

.PHONY: test-slow
test-slow: ## Run only the tests marked `slow`
	$(PYTHON) -m pytest -m slow

.PHONY: test-all
test-all: ## Run every test, including the slow ones
	$(PYTHON) -m pytest -m ""

.PHONY: coverage
coverage: ## Run the suite with coverage and enforce the thresholds
	$(PYTHON) -m pytest --cov=nas_engine --cov-report=term-missing --cov-report=xml
	@echo ""
	$(PYTHON) scripts/check_coverage.py

# ------------------------------------------------------------------------------ usage --
.PHONY: smoke
smoke: ## Run a real end-to-end search through the CLI
	scripts/run_smoke_search.sh $(SMOKE_OUTPUT)

.PHONY: examples
examples: ## Run every example script
	$(PYTHON) examples/quickstart.py
	$(PYTHON) examples/custom_search_space.py
	$(PYTHON) examples/custom_objective.py
	$(PYTHON) examples/resume_search.py

.PHONY: benchmark
benchmark: ## Run the micro-benchmarks
	$(PYTHON) scripts/benchmark.py

.PHONY: docs
docs: ## Check every documentation link, anchor, table, and the manifest
	$(PYTHON) scripts/check_docs_links.py
	$(PYTHON) scripts/check_tables.py
	$(PYTHON) scripts/generate_manifest.py --check

.PHONY: docs-fix
docs-fix: ## Reformat documentation tables and regenerate the manifest
	$(PYTHON) scripts/check_tables.py --fix
	$(PYTHON) scripts/generate_manifest.py

.PHONY: manifest
manifest: ## Regenerate the source tables in docs/repository-manifest.md
	$(PYTHON) scripts/generate_manifest.py

# ---------------------------------------------------------------------------- packaging --
.PHONY: build
build: clean-build ## Build the source distribution and wheel
	$(PYTHON) -m build

.PHONY: verify-package
verify-package: build ## Install the built wheel into a throwaway environment and smoke it
	@set -euo pipefail; \
	tmp=$$(mktemp -d); \
	trap 'rm -rf "$$tmp"' EXIT; \
	$(PYTHON) -m venv "$$tmp/venv"; \
	"$$tmp/venv/bin/pip" install --quiet --upgrade pip; \
	"$$tmp/venv/bin/pip" install --quiet dist/*.whl; \
	"$$tmp/venv/bin/nas-engine" --help > /dev/null; \
	"$$tmp/venv/bin/python" -c "import nas_engine; print('installed', nas_engine.__version__)"; \
	echo "the built wheel installs and runs in a clean environment"

.PHONY: docker-build
docker-build: ## Build the container image
	docker build -t nas-engine:local .

.PHONY: docker-smoke
docker-smoke: docker-build ## Run the smoke search inside the container
	docker run --rm nas-engine:local smoke

# ------------------------------------------------------------------------------ cleanup --
.PHONY: clean-build
clean-build: ## Remove build artefacts
	rm -rf build dist src/*.egg-info

.PHONY: clean
clean: clean-build ## Remove build artefacts, caches, and generated outputs
	rm -rf .pytest_cache .mypy_cache .ruff_cache .hypothesis htmlcov
	rm -f coverage.xml .coverage .coverage.*
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find artifacts -mindepth 1 ! -name .gitkeep -prune -exec rm -rf {} + 2>/dev/null || true
