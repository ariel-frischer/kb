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
uv tool install "kb[all] @ /home/ari/repos/kb" --force   # install as `kb` command from local
uv tool install "kb[all] @ git+https://github.com/ariel-frischer/kb.git"  # install from git
```

## Scope Model

Two modes — global (default) and project-local:

- **Global**: config at `~/.config/kb/config.toml`, DB at `~/.local/share/kb/kb.db`. Sources are absolute paths. `kb init` creates global config.
- **Project**: config at `.kb.toml` (walk-up from cwd), DB at `~/.local/share/kb/projects/<hash>/kb.db`. Sources are relative paths. `kb init --project` creates project config. No files stored in the project directory (no `.kb/` dir).

Project `.kb.toml` takes precedence over global config when both exist. Config walk-up works like `.gitignore` — `kb` works from any subdirectory. DB path is deterministic via SHA-256 hash of the resolved config directory (`_project_db_path()`). If a `.kb.toml` explicitly sets `db = "..."`, that path is used instead (backward compat).

Path resolution: `Config.doc_path_for_db()` computes stored paths — relative to config_dir in project mode, `source_dir.name/relative` in global mode.

## Architecture

- `cli.py` — entry point, command dispatch via `sys.argv` (no argparse). Each `cmd_*` is a thin wrapper that calls the corresponding `_core` function from `api.py` and handles printing/exit.
- `api.py` — core logic for search/ask/fts/similar/stats/list. Returns plain dicts, never prints or calls `sys.exit()`. Used by both CLI and MCP server. Exception classes: `KBError`, `NoIndexError`, `NoSearchTermsError`, `FileNotIndexedError`.
- `mcp_server.py` — MCP server (FastMCP, stdio transport) exposing 6 tools: `kb_search`, `kb_ask`, `kb_fts`, `kb_similar`, `kb_status`, `kb_list`. Calls `api.py` core functions. Optional dep (`mcp[cli]`).
- `config.py` — `Config` dataclass with all tunables, `find_config()` walk-up loader, `save_config()` minimal TOML serializer (only writes non-default values)
- `extract.py` — text extraction registry. `_register()` maps extensions to `(extractor_fn, doc_type, available, install_hint, is_code)`. Stdlib formats always available; optional deps (pymupdf, python-docx, etc.) probed at import time.
- `ingest.py` — indexing pipeline: discover files → `.kbignore` filtering → size guard → `extract_text()` → frontmatter tag parsing (markdown) → content-hash diff → chunk → diff chunks by hash → batch embed new → store
- `db.py` — schema creation + `SCHEMA_VERSION` migration. Tables: `documents` (with `tags` column), `chunks` (with `doc_path` + `fts_path` columns), `vec_chunks` (vec0 virtual table with cosine distance), `fts_chunks` (FTS5 with 3 columns: doc_path/heading/text, weighted BM25, porter stemming, trigger-based sync using `fts_path`). WAL mode + foreign keys enabled. `fts_path()` returns last 2 path components to avoid IDF collapse from common prefixes. v8→v9 switches vec0 to cosine distance (drops vec_chunks, requires reindex); v7→v8 adds `fts_path` + rebuilds FTS with truncated paths; v6→v7 adds `doc_path` to chunks + FTS with field weights; v5→v6 rebuilds FTS with porter tokenizer; v4→v5 rebuilds FTS with triggers; v3→v4 uses ALTER TABLE; older versions drop-and-recreate.
- `chunk.py` — markdown (heading-aware with ancestry tracking) + plain text chunking. Uses chonkie with overlap refinery when available, regex fallback otherwise. `embedding_text()` enriches chunks with file path + heading ancestry before embedding.
- `search.py` — hybrid search: vector (vec0 MATCH) + FTS5 (AND + prefix matching), fused with score-weighted RRF (vec scaled by similarity, FTS by normalized BM25) with positional rank bonuses. `fill_fts_only_results()` backfills metadata for FTS-only hits. `filter_vec_by_ids()` / `filter_fts_by_ids()` for pre-fusion tag filtering.
- `rerank.py` — Reranking dispatcher: `rerank()` routes to `cross_encoder_rerank()` (local, sentence-transformers) or `llm_rerank()` (RankGPT pattern) based on `cfg.rerank_method`. Cross-encoder uses GPU if available (CUDA > MPS > CPU), lazy-loads and caches model. Returns `(results, rerank_info)` tuple.
- `hyde.py` — HyDE (Hypothetical Document Embeddings). `generate_hyde_passage()` dispatches to `local_hyde_passage()` (causal LM via transformers) or `llm_hyde_passage()` (OpenAI API) based on `cfg.hyde_method`. Local method uses lazy-loaded model cache (`_hyde_model_cache`), bf16 on GPU, same pattern as expand.py. Returns `(passage, elapsed_ms)`, `None` on failure (graceful fallback to raw query).
- `expand.py` — Query expansion dispatcher: `expand_query()` routes to `local_expand()` (FLAN-T5-small via transformers, lazy-loaded and cached) or `llm_expand()` (OpenAI JSON mode). Returns typed expansions `[{"type": "lex"|"vec", "text": "..."}]`. Filters duplicates of original query. Graceful fallback to `[]` on error.
- `filters.py` — inline filter syntax (`file:`, `type:`, `tag:`, `dt>`, `dt<`, `+"kw"`, `-"kw"`) parsed from query string. `get_tagged_chunk_ids()` returns chunk IDs for tag-matching documents (used for pre-fusion filtering). `remove_tag_filter()` strips tag from post-filters after pre-fusion handling.
- `embed.py` — thin OpenAI embedding wrapper, `serialize_f32()` / `deserialize_f32()` for sqlite-vec binary format

### Data Flow

**Index**: files → extract_text → content-hash check (skip unchanged) → chunk → diff chunks by hash (reuse unchanged) → compute `fts_path` (last 2 path components) → batch embed new → store in vec0 (FTS5 synced via triggers using `fts_path`)

**Search**: query → parse_filters → [HyDE best-of-two: embed both raw query + HyDE passage in one batch, run two vec queries, keep set with better top-1 similarity] → [expand] → vec0 MATCH + FTS5 MATCH (original + expansion queries) → pre-filter by tagged chunk IDs if `tag:` active → multi-list weighted RRF (primary 2.0, expansions 1.0) → apply remaining filters → display. `fused` count captured pre-filter; `after_filters` added when filters active.

**FTS**: query → parse_filters → FTS5 MATCH → weighted BM25 scores (fts_path 10x, heading 2x, text 1x) → apply_filters → display (no embedding, instant). `fts_path` stores last 2 path components to avoid IDF collapse from common prefixes.

**Ask**: question → parse_filters → BM25 probe (LIMIT 20, deduplicate by document, shortcut if top-doc norm >= `bm25_shortcut_min` with gap >= `bm25_shortcut_gap` vs second-doc) → if shortcut: FTS only; else: [HyDE best-of-two] → [expand] → hybrid search → pre-filter by tagged chunk IDs if `tag:` active → multi-list weighted RRF → apply remaining filters → rerank (cross-encoder or LLM) → top rerank_top_k → confidence threshold → LLM generates answer.

**Similar**: resolve file → read chunk embeddings from vec0 → average into doc vector → KNN query → filter out source doc → aggregate best distance per doc → display

### Key Design Decisions

- **vec0 with cosine distance** — `vec_chunks` uses `distance_metric=cosine`, so distances are in [0, 2] and `1 - distance` gives true cosine similarity in [-1, 1]. Auxiliary columns store chunk_text, doc_path, heading alongside embeddings, avoiding JOINs at search time
- **Content-hash at two levels** — file-level hash skips unchanged files entirely; chunk-level hash avoids re-embedding unchanged chunks within modified files
- **FTS5 field-weighted trigger sync** — `fts_chunks` has 3 columns (`doc_path`, `heading`, `text`) with weighted BM25 (`bm25(10.0, 2.0, 1.0)`) set via rank config. Uses `content='chunks'` with INSERT/DELETE/UPDATE triggers (`fts_ai`, `fts_ad`, `fts_au`) for automatic sync. Triggers read `fts_path` (last 2 path components) instead of full `doc_path` to avoid IDF collapse from common prefixes. Porter stemming (`tokenize='porter unicode61'`) for word-form matching. Filepath matches get 10x weight, headings 2x
- **Score-weighted RRF with rank bonuses** — fusion weights vec results by `similarity / (k + rank) + bonus` and FTS by `norm_bm25 / (k + rank) + bonus` where `norm_bm25 = |score| / (1 + |score|)` and bonus is +0.05 for rank 0, +0.02 for ranks 1-2
- **HyDE best-of-two** — before vector search, generates a hypothetical answer passage (~100-200 words), then embeds BOTH the raw query and HyDE passage in a single batch API call, runs two vec queries, and keeps whichever result set has better top-1 similarity. HyDE can only help, never hurt. FTS still uses original query keywords. Enabled by default (`hyde_enabled = true`). Two methods: `"llm"` (OpenAI API, uses `hyde_model` or `chat_model`) or `"local"` (causal LM via transformers, default `Qwen/Qwen3-0.6B`, no API cost). Local method uses lazy-loaded model cache with bf16 on GPU if available. Falls back to raw query only when passage generation fails.
- **Query expansion** — opt-in (`query_expand = false` by default, `--expand` CLI flag). Generates typed expansions: `lex` (keyword synonyms for FTS) and `vec` (semantic rephrasings for vector search). Two methods: `local` (FLAN-T5-small via transformers, lazy-loaded/cached, ~1s CPU) or `llm` (OpenAI JSON mode via `cfg.chat_model`). Expansion results fused with primary results via `multi_rrf_fuse()` — primary lists weighted 2.0, expansion lists 1.0. BM25 shortcut skips expansion entirely.
- **BM25 shortcut in ask** — probes top 20 FTS results, deduplicates by document (keeps best norm per doc), compares top-doc vs second-doc; if top norm >= `bm25_shortcut_min` (default 0.85) with gap >= `bm25_shortcut_gap` (default 0.05), skips HyDE/expansion/embedding/vector/rerank and uses FTS results directly. Both thresholds configurable. Document-level dedup prevents same-doc chunks from suppressing the gap.
- **Tag pre-retrieval filtering** — when `tag:` filter is active, `get_tagged_chunk_ids()` computes the set of chunk IDs belonging to tag-matching documents. Vec and FTS results are filtered to this set *before* RRF fusion (`filter_vec_by_ids()` / `filter_fts_by_ids()`), then tag is removed from post-filters. `retrieve_k` is inflated to `len(tagged_ids) + top_k` to ensure enough candidates
- **Schema versioning** — `SCHEMA_VERSION` in `meta` table; v8→v9 drops vec_chunks and recreates with cosine distance (requires reindex); v7→v8 adds `fts_path` to chunks + rebuilds FTS with truncated paths (manual rebuild, not FTS5 built-in, because `content='chunks'` maps to `doc_path`); v6→v7 adds `doc_path` to chunks + FTS with field weights; v5→v6 rebuilds FTS with porter tokenizer; v4→v5 rebuilds FTS with triggers; v3→v4 uses non-destructive ALTER TABLE; older upgrades drop and recreate all tables
- **Structured output** — `search`, `fts`, and `ask` support `--json`, `--csv`, and `--md` flags for structured output (JSON for scripting/agents, CSV for spreadsheets, markdown tables for docs/LLMs)
- **Tags** — stored comma-separated in `documents.tags` column; auto-parsed from markdown YAML frontmatter during indexing; manually managed via `kb tag`/`kb untag`
- **MCP server** — `kb mcp` / `kb-mcp` starts a Model Context Protocol server (stdio transport) exposing search/ask/fts/similar/status/list as tools for Claude Desktop, Claude Code, and other MCP clients. Optional dep: `mcp[cli]` (included in `kb[all]`)
- **CLI/API split** — `api.py` contains all core logic (returns dicts), `cli.py` is a thin presentation layer. Both CLI and MCP server call the same core functions.
- **Structured changelog** — `CHANGELOG.yaml` with per-release categories (added/fixed/changed/removed/deprecated/breaking), commit SHAs, migration notes. `make changelog` scaffolds entries from git history via `scripts/changelog_entry.py`.
