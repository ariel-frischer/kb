# CLAUDE.md

## Project: kb

CLI knowledge base tool. Indexes markdown + PDFs, hybrid search (sqlite-vec + FTS5), RAG answers with LLM rerank.

## Build & Test

```bash
uv sync --all-extras        # install with all optional deps
uv run kb --help            # run locally
uv run pytest               # run tests
uv run ruff check .         # lint
uv run ruff format .        # format
```

## Install globally

```bash
uv tool install ".[all]" --from .   # install as `kb` command
```

## Scope Model

Two modes — global (default) and project-local:

- **Global**: config at `~/.config/kb/config.toml`, DB at `~/.local/share/kb/kb.db`. Sources are absolute paths. `kb init` creates global config.
- **Project**: config at `.kb.toml` (walk-up from cwd), DB next to config. Sources are relative paths. `kb init --project` creates project config.

Project `.kb.toml` takes precedence over global config when both exist.

Source management: `kb add <dir>`, `kb remove <dir>`, `kb sources`.

## Architecture

- `src/kb/cli.py` — entry point, command dispatch (init, add, remove, sources, index, search, ask, stats, reset)
- `src/kb/config.py` — config loading (project .kb.toml + global ~/.config/kb/config.toml), Config dataclass, save_config
- `src/kb/db.py` — schema, sqlite-vec connection
- `src/kb/chunk.py` — markdown + plain text chunking (chonkie or regex fallback)
- `src/kb/embed.py` — OpenAI embedding helpers
- `src/kb/search.py` — hybrid search, RRF fusion
- `src/kb/rerank.py` — LLM reranking (RankGPT pattern)
- `src/kb/filters.py` — pre-search filter parsing + application
- `src/kb/ingest.py` — file indexing (markdown + PDF), uses Config.doc_path_for_db() for path resolution
