# Repository Guidelines

## Project Structure & Module Organization

This is a Python 3.12+ CLI package using the `src` layout. Core code lives in `src/kb/`: `cli.py` handles command dispatch, `api.py` contains command logic, and modules such as `db.py`, `search.py`, `ingest.py`, `embed.py`, and `config.py` own focused subsystems. Tests live in `tests/` and generally mirror feature modules, for example `tests/test_search.py`. Developer docs are in `docs/`, release metadata is in `CHANGELOG.yaml`, and utility scripts are in `scripts/`.

## Build, Test, and Development Commands

Use `uv` for environment and command execution.

- `make install-dev`: sync all extras and dev tools with `uv sync --all-extras`.
- `make test`: run the full pytest suite.
- `make lint`: run `ruff check .`.
- `make format`: format code with `ruff format .`.
- `make check`: run the CI-equivalent lint, format check, and tests.
- `uv run kb --help`: run the local CLI entry point.
- `make changelog-write`: scaffold and insert a structured changelog entry.

## Coding Style & Naming Conventions

Prefer small, readable functions with flat control flow and explicit names. Keep CLI wrappers thin: add reusable behavior to `api.py`, then call it from `cli.py`. Use Ruff for formatting and linting. Python modules and functions use `snake_case`, tests use `test_*.py`, and public CLI commands should have matching `cmd_<name>()` wrappers plus `<name>_core()` functions when reusable logic is exposed.

## Testing Guidelines

Tests use `pytest`. Add or update focused tests for behavior changes, especially search ranking, filtering, ingestion, config loading, database migrations, and CLI output. Run `make test` locally and `make check` before opening a PR. For targeted iteration, use `uv run pytest tests/test_filters.py -q`.

## Commit & Pull Request Guidelines

Git history uses conventional commit prefixes such as `feat:`, `fix:`, `docs:`, `refactor:`, `style:`, and `chore:`. Keep commits scoped and mention affected modules when useful, for example `fix(search): handle empty FTS results`.

PRs should include a clear description, tests run, linked issues when applicable, and CLI output examples for user-visible command changes. Update `CHANGELOG.yaml` for release-relevant changes.

## Security & Configuration Tips

Do not commit secrets, API keys, local databases, or generated cache files. Project configuration is loaded from `.kb.toml` by walking up from the current directory, while user secrets belong in local config files outside version control. Be careful with changes that affect embedding providers, database paths, or indexing defaults because they can trigger API costs or reindexing work.
