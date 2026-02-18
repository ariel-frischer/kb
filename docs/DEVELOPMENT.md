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
├── cli.py         — Entry point, command dispatch, human-readable output (thin wrappers over api.py)
├── api.py         — Core logic for search/ask/fts/similar/stats/list/feedback (returns dicts, no I/O)
├── mcp_server.py  — MCP server (FastMCP, stdio) exposing kb tools for AI agents
├── config.py      — .kb.toml loading, Config dataclass, secrets.toml loading
├── db.py          — SQLite schema, sqlite-vec connection, migrations
├── chunk.py       — Markdown + plain text chunking (chonkie or regex fallback)
├── embed.py       — OpenAI embedding helpers, batching, serialize/deserialize for sqlite-vec
├── extract.py     — Text extraction registry for 30+ formats (PDF, DOCX, EPUB, HTML, ODT, etc.)
├── hyde.py        — HyDE: generates hypothetical answer passage (local model or LLM API) for better vector retrieval
├── expand.py      — Query expansion: local (FLAN-T5) or LLM, generates keyword + semantic variants
├── search.py      — Hybrid search (vector + FTS5), RRF fusion, multi-list RRF for expansion
├── rerank.py      — Reranking: local cross-encoder (sentence-transformers) or LLM (RankGPT)
├── filters.py     — Pre-search filter parsing + application (file:, type:, tag:, dt>, dt<, +"kw", -"kw")
└── ingest.py      — File indexing pipeline (unified loop over all supported formats, frontmatter tag parsing)
```

### Data flow

**Indexing** (`kb index`): find files by extension → extract text (format-specific) → chunking → content-hash diff → embed new chunks → store in sqlite-vec (vec0) + FTS5

**Search** (`kb search`): query → parse filters → [HyDE best-of-two: embed raw query + passage, keep better vec results] → [expand] → vector search (vec0 cosine) + FTS5 (original + expansion queries; SQL-level pre-filtered to tagged chunk IDs if `tag:` active) → multi-list weighted RRF (primary 2x, expansions 1x) → apply remaining filters → results

**Ask** (`kb ask`): BM25 probe (LIMIT 20, dedup by document, shortcut if top norm >= `bm25_shortcut_min` with gap >= `bm25_shortcut_gap`) → if shortcut: FTS only; else: [HyDE best-of-two] → [expand] → vec+fts (multi-query; SQL-level pre-filtered to tagged chunk IDs if `tag:` active) → multi-list weighted RRF → apply remaining filters → rerank (cross-encoder or LLM) → confidence threshold → LLM generates answer from context

**Similar** (`kb similar`): read chunk embeddings from vec0 → average into doc vector → KNN query → filter self → aggregate by doc → rank by similarity

### Key design decisions

- **sqlite-vec `vec0` with cosine distance** — stores embeddings + text in auxiliary columns, avoiding JOINs at search time. Uses `distance_metric=cosine` so `1 - distance` gives true cosine similarity
- **Reciprocal Rank Fusion** — combines vector and keyword rankings without needing score normalization
- **FTS5 field weighting** — `fts_path` (10x), `heading` (2x), `text` (1x) via BM25 rank config. `fts_path` stores last 2 path components to avoid IDF collapse from common prefixes; filepath matches strongly boost relevance
- **HyDE best-of-two** — embeds both raw query and hypothetical passage in one batch, runs two vec queries, keeps whichever has better top-1 similarity. HyDE can only help, never hurt. Two methods: `"llm"` (OpenAI API) or `"local"` (causal LM via transformers, default Qwen/Qwen3-0.6B, no API cost). FTS still uses original query.
- **Query expansion** — opt-in (`--expand`), generates keyword synonyms (`lex`) and semantic rephrasings (`vec`) via local FLAN-T5 or LLM, fused with primary results via multi-list weighted RRF
- **Content-hash per chunk** — incremental indexing only re-embeds changed content
- **Config walks up from cwd** — like `.gitignore`, so `kb` works from any subdirectory
- **Project DB in XDG data dir** — project-mode databases live at `~/.local/share/kb/projects/<hash>/kb.db` (SHA-256 of config dir), keeping WAL sidecar files out of the project directory. Explicit `db = "..."` in `.kb.toml` overrides for backward compat
- **Tags** — comma-separated in `documents.tags` column; auto-parsed from markdown YAML frontmatter, manually managed via `kb tag`/`kb untag`

## Changelog

The project keeps a structured `CHANGELOG.yaml` at the repo root. Each release entry has categories (`added`, `fixed`, `changed`, `removed`, `deprecated`, `breaking`), commit SHAs, and optional migration notes.

```bash
make changelog          # preview new entry from commits since last tag
make changelog-write    # insert entry into CHANGELOG.yaml
```

The scaffold script (`scripts/changelog_entry.py`) reads the version from `pyproject.toml`, collects commits since the last tag, and categorizes by conventional commit prefix. Review and edit the generated entry before committing — auto-generated descriptions are a starting point, not final copy.

## Reference docs

- [.kbignore patterns](kbignore.md) — common ignore patterns by use case

## Adding a new command

1. Add a `<name>_core()` function in `api.py` that returns a dict (no printing, no `sys.exit()`)
2. Add a `cmd_<name>()` wrapper in `cli.py` that calls the core function and handles output/errors
3. Add dispatch in `main()` — before `find_config()` if no config needed (like `completion`), in the `elif` chain otherwise
4. Add a help line in the `USAGE` string
5. Optionally expose as an MCP tool in `mcp_server.py`

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
5. Update `CHANGELOG.yaml` (add entry under `unreleased`)
6. Open a PR against `main`
