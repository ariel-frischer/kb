# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project: kb

CLI knowledge base tool. Indexes 30+ document formats (markdown, PDF, DOCX, EPUB, HTML, ODT, RTF, plain text, email, subtitles, and more). Hybrid search (sqlite-vec + FTS5), RAG answers with LLM rerank.

## Build & Test

```bash
uv sync --all-extras          # install with all optional deps
uv run kb --help              # run locally
uv run pytest                 # run all tests
uv run pytest tests/test_chunk.py -v              # single test file
uv run pytest tests/test_chunk.py::test_name -v   # single test
uv run ruff check .           # lint
uv run ruff format .          # format
make check                    # lint + format check + tests (CI equivalent)
```

## Install globally

```bash
uv tool install ".[all]" --from .   # install as `kb` command
```

## Scope Model

Two modes — global (default) and project-local:

- **Global**: config at `~/.config/kb/config.toml`, DB at `~/.local/share/kb/kb.db`. Sources are absolute paths. `kb init` creates global config.
- **Project**: config at `.kb.toml` (walk-up from cwd), DB next to config. Sources are relative paths. `kb init --project` creates project config.

Project `.kb.toml` takes precedence over global config when both exist. Config walk-up works like `.gitignore` — `kb` works from any subdirectory.

Path resolution: `Config.doc_path_for_db()` computes stored paths — relative to config_dir in project mode, `source_dir.name/relative` in global mode.

## Architecture

- `cli.py` — entry point, command dispatch via `sys.argv` (no argparse). Each `cmd_*` function handles one command.
- `config.py` — `Config` dataclass with all tunables, `find_config()` walk-up loader, `save_config()` minimal TOML serializer (only writes non-default values)
- `extract.py` — text extraction registry. `_register()` maps extensions to `(extractor_fn, doc_type, available, install_hint, is_code)`. Stdlib formats always available; optional deps (pymupdf, python-docx, etc.) probed at import time.
- `ingest.py` — indexing pipeline: discover files → `.kbignore` filtering → size guard → `extract_text()` → frontmatter tag parsing (markdown) → content-hash diff → chunk → diff chunks by hash → batch embed new → store
- `db.py` — schema creation + `SCHEMA_VERSION` migration. Tables: `documents` (with `tags` column), `chunks`, `vec_chunks` (vec0 virtual table), `fts_chunks` (FTS5 content-sync'd from chunks). v3→v4 uses ALTER TABLE (non-destructive); older versions drop-and-recreate.
- `chunk.py` — markdown (heading-aware with ancestry tracking) + plain text chunking. Uses chonkie with overlap refinery when available, regex fallback otherwise. `embedding_text()` enriches chunks with file path + heading ancestry before embedding.
- `search.py` — hybrid search: vector (vec0 MATCH) + FTS5, fused with Reciprocal Rank Fusion. `fill_fts_only_results()` backfills metadata for FTS-only hits.
- `rerank.py` — RankGPT pattern: presents numbered passages to LLM, parses comma-separated ranking response
- `filters.py` — inline filter syntax (`file:`, `type:`, `tag:`, `dt>`, `dt<`, `+"kw"`, `-"kw"`) parsed from query string, applied post-search
- `embed.py` — thin OpenAI embedding wrapper, `serialize_f32()` / `deserialize_f32()` for sqlite-vec binary format

### Data Flow

**Index**: files → extract_text → content-hash check (skip unchanged) → chunk → diff chunks by hash (reuse unchanged) → batch embed new → store in vec0 + rebuild FTS5

**Search**: query → parse_filters → embed → vec0 MATCH + FTS5 MATCH → RRF fusion → apply_filters → display

**Ask**: same as search but over-fetches (rerank_fetch_k=20) → LLM rerank → top rerank_top_k → confidence threshold (min_similarity) → LLM generates answer

**Similar**: resolve file → read chunk embeddings from vec0 → average into doc vector → KNN query → filter out source doc → aggregate best distance per doc → display

### Key Design Decisions

- **vec0 auxiliary columns** — `vec_chunks` stores chunk_text, doc_path, heading alongside embeddings, avoiding JOINs at search time
- **Content-hash at two levels** — file-level hash skips unchanged files entirely; chunk-level hash avoids re-embedding unchanged chunks within modified files
- **FTS5 content-sync** — `fts_chunks` uses `content='chunks'` with manual rebuild after indexing
- **Schema versioning** — `SCHEMA_VERSION` in `meta` table; v3→v4 uses non-destructive ALTER TABLE; older upgrades drop and recreate all tables
- **Tags** — stored comma-separated in `documents.tags` column; auto-parsed from markdown YAML frontmatter during indexing; manually managed via `kb tag`/`kb untag`
