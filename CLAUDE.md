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

## Architecture

- `src/kb/cli.py` — entry point, command dispatch
- `src/kb/config.py` — .kb.toml loading, Config dataclass
- `src/kb/db.py` — schema, sqlite-vec connection
- `src/kb/chunk.py` — markdown + plain text chunking (chonkie or regex fallback)
- `src/kb/embed.py` — OpenAI embedding helpers
- `src/kb/search.py` — hybrid search, RRF fusion
- `src/kb/rerank.py` — LLM reranking (RankGPT pattern)
- `src/kb/filters.py` — pre-search filter parsing + application
- `src/kb/ingest.py` — file indexing (markdown + PDF)
