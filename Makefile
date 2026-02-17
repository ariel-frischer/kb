.PHONY: install install-dev lint format test check clean changelog

install:  ## Install with all features
	uv tool install "kb[all] @ ." --force --reinstall

install-dev:  ## Install dev environment
	uv sync --all-extras

lint:  ## Run linter
	uv run ruff check .

format:  ## Format code
	uv run ruff format .

test:  ## Run tests
	uv run pytest

check: lint  ## Lint + format check + test
	uv run ruff format --check .
	uv run pytest

changelog:  ## Scaffold changelog entry from recent commits
	uv run python scripts/changelog_entry.py

changelog-write:  ## Scaffold and insert changelog entry into CHANGELOG.yaml
	uv run python scripts/changelog_entry.py --write

clean:  ## Remove build artifacts
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/ .ruff_cache/

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
