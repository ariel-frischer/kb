# kb

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

CLI knowledge base: index markdown + PDFs, hybrid search (semantic + keyword), RAG answers. Powered by [sqlite-vec](https://github.com/asg017/sqlite-vec).

## Install

```bash
# From PyPI (with all features)
pip install "kb[all]"

# Or with uv
uv tool install "kb[all]"

# From source
git clone https://github.com/arielfrischer/kb.git
cd kb
uv tool install ".[all]"
```

Requires `OPENAI_API_KEY` in your environment (or in `~/.config/kb/secrets.yaml`).

## Quickstart

```bash
# 1. Create config in your project
cd ~/my-project
kb init

# 2. Edit .kb.toml — add your source directories
#    sources = ["docs/", "notes/"]

# 3. Index
kb index

# 4. Search
kb search "deployment patterns"

# 5. Ask (RAG: search + LLM rerank + answer)
kb ask "what are the recommended deployment patterns?"

# 6. Check what's indexed
kb stats
```

## Commands

```
kb init                   Create .kb.toml in current directory
kb index [DIR...]         Index sources from config (or explicit dirs)
kb search "query" [k]     Hybrid search (default k=5)
kb ask "question" [k]     RAG answer (default k=8)
kb stats                  Show index stats + capabilities
kb reset                  Drop DB and start fresh
```

## .kb.toml

Created by `kb init`. Config is found by walking up from cwd (like `.gitignore`).

```toml
# Where to store the database (relative to this file)
db = "kb.db"

# Directories to index (relative to this file)
sources = [
    "docs/",
    "notes/",
]

# All optional — defaults shown
# embed_model = "text-embedding-3-small"
# embed_dims = 1536
# chat_model = "gpt-4o-mini"
# max_chunk_chars = 2000
# min_similarity = 0.25
# rerank_fetch_k = 20
# rerank_top_k = 5
```

## .kbignore

Drop a `.kbignore` in any source directory. Gitignore-style syntax:

```
# Skip directories
internal/
drafts/

# Skip file patterns
*.draft.md
WIP-*
```

## Search Filters

Add inline with your query — stripped before embedding:

```bash
kb search 'file:articles/*.md cost optimization'
kb search 'dt>"2026-02-01" recent developments'
kb search '+"docker" -"kubernetes" container setup'
kb ask 'file:briefs/*.pdf dt>"2026-02-13" what are the costs?'
```

| Filter | Syntax | Example |
|---|---|---|
| File glob | `file:<pattern>` | `file:articles/*.md` |
| After date | `dt>"YYYY-MM-DD"` | `dt>"2026-02-01"` |
| Before date | `dt<"YYYY-MM-DD"` | `dt<"2026-02-14"` |
| Must contain | `+"keyword"` | `+"docker"` |
| Must not contain | `-"keyword"` | `-"kubernetes"` |

## Features

### Always on

- **Hybrid search** — vector similarity + FTS5 keyword search, fused with Reciprocal Rank Fusion
- **Heading-aware chunking** — markdown split by heading hierarchy, each chunk carries ancestry
- **Incremental indexing** — content-hash per chunk, only re-embeds changes
- **Auxiliary columns** — vec0 stores text alongside vectors, no JOINs at search time
- **Confidence threshold** — `ask` filters low-similarity results before LLM
- **LLM rerank** — `ask` over-fetches 20 results, LLM ranks by relevance, keeps top 5
- **Pre-search filters** — inline filter syntax in queries

### Auto-enabled with extras

| Feature | Install | Without it |
|---|---|---|
| chonkie chunking | `kb[chunking]` or `kb[all]` | Falls back to regex splitter |
| PDF ingestion | `kb[pdf]` or `kb[all]` | PDFs skipped |

## How It Works

```
kb index
  1. Find .md + .pdf files (respecting .kbignore)
  2. Content-hash check — skip unchanged files
  3. Chunk (chonkie or regex fallback)
  4. Diff chunks by hash — only embed new/changed
  5. Batch embed via OpenAI
  6. Store in sqlite-vec (vec0) + FTS5

kb search "query"
  1. Parse filters, strip from query
  2. Embed clean query
  3. Vector search (vec0 MATCH) + FTS5 keyword search
  4. Fuse with RRF
  5. Apply filters
  6. Display results

kb ask "question"
  1. Same as search, but over-fetch 20
  2. Apply filters
  3. LLM rerank -> top 5
  4. Confidence threshold
  5. LLM generates answer from context
```

## Development

```bash
git clone https://github.com/arielfrischer/kb.git
cd kb
uv sync --all-extras
uv run pytest
uv run ruff check .
```

## Contributing

Contributions welcome! Please open an issue first to discuss what you'd like to change.

1. Fork the repo
2. Create a feature branch (`git checkout -b feat/my-feature`)
3. Make your changes and add tests
4. Run `uv run pytest && uv run ruff check .`
5. Open a PR

## License

[MIT](LICENSE)
