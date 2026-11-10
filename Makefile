PY ?= python
PYTEST ?= $(PY) -m pytest
RUFF ?= $(PY) -m ruff

.PHONY: install test test-fast test-gpu lint format cov clean

install:
	$(PY) -m pip install -e ".[dev]"

test:
	$(PYTEST) -q -m "not gpu"

test-fast:
	$(PYTEST) -q -m "not gpu and not slow"

test-gpu:
	$(PYTEST) -q -m gpu

cov:
	$(PYTEST) -q -m "not gpu" --cov=frolic --cov-report=term-missing --cov-report=xml

lint:
	$(RUFF) check frolic tests
	$(RUFF) format --check frolic tests

format:
	$(RUFF) format frolic tests
	$(RUFF) check --fix frolic tests

clean:
	rm -rf build dist .pytest_cache .ruff_cache htmlcov coverage.xml .coverage
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
