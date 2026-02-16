# Development Guide

## Setup

```bash
git clone https://github.com/arielfrischer/kb.git
cd kb
uv sync --all-extras    # install with all optional deps + dev tools
```

## Common Commands

```bash
make help               # show all available commands
make check              # lint + format check + tests (CI equivalent)
make test               # run tests only
make lint               # run linter only
make format             # auto-format code
```

Or without make:

```bash
uv run pytest           # run tests
uv run ruff check .     # lint
uv run ruff format .    # format
uv run kb --help        # run locally
```

## Architecture

```
src/kb/
├── cli.py       — Entry point, command dispatch (init, index, search, ask, stats, reset)
├── config.py    — .kb.toml loading, Config dataclass, secrets.toml loading
├── db.py        — SQLite schema, sqlite-vec connection, migrations
├── chunk.py     — Markdown + plain text chunking (chonkie or regex fallback)
├── embed.py     — OpenAI embedding helpers, batching
├── extract.py   — Text extraction registry for 30+ formats (PDF, DOCX, EPUB, HTML, ODT, etc.)
├── search.py    — Hybrid search (vector + FTS5), RRF fusion
├── rerank.py    — LLM reranking (RankGPT pattern)
├── filters.py   — Pre-search filter parsing + application
└── ingest.py    — File indexing pipeline (unified loop over all supported formats)
```

### Data flow

**Indexing** (`kb index`): find files by extension → extract text (format-specific) → chunking → content-hash diff → embed new chunks → store in sqlite-vec (vec0) + FTS5

**Search** (`kb search`): query → parse filters → embed → vector search + FTS5 → RRF fusion → apply filters → results

**Ask** (`kb ask`): same as search but over-fetches → LLM reranks top candidates → confidence threshold → LLM generates answer from context

### Key design decisions

- **sqlite-vec `vec0` virtual table** — stores embeddings + text in auxiliary columns, avoiding JOINs at search time
- **Reciprocal Rank Fusion** — combines vector and keyword rankings without needing score normalization
- **Content-hash per chunk** — incremental indexing only re-embeds changed content
- **Config walks up from cwd** — like `.gitignore`, so `kb` works from any subdirectory

## Reference docs

- [.kbignore patterns](kbignore.md) — common ignore patterns by use case

## Adding a new command

1. Add the function in `cli.py` following the existing pattern (parse args → load config → connect → execute)
2. Register it in the `COMMANDS` dict at the bottom of `cli.py`
3. Add a help line in `usage()`

## Running tests

```bash
make test
# or
uv run pytest -v
```

## PR workflow

1. Fork the repo
2. Create a feature branch (`git checkout -b feat/my-feature`)
3. Make changes and add tests
4. Run `make check` (must pass)
5. Open a PR against `main`
