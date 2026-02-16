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

Requires an OpenAI-compatible API. Set `OPENAI_API_KEY` in your environment (or in `~/.config/kb/secrets.toml`).

Works with any provider that speaks the OpenAI API — set `OPENAI_BASE_URL` to point at Ollama, LiteLLM, vLLM, etc.

## Quickstart

```bash
# 1. Initialize (global — indexes across repos/folders)
kb init

# 2. Add source directories
kb add ~/notes ~/docs ~/repos/my-project/docs

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
kb init                   Create global config (~/.config/kb/)
kb init --project         Create project-local .kb.toml in current directory
kb add <dir> [dir...]     Add source directories
kb remove <dir> [dir...]  Remove source directories
kb sources                List configured sources
kb index [DIR...]         Index sources from config (or explicit dirs)
kb search "query" [k]     Hybrid search (default k=5)
kb ask "question" [k]     RAG answer (default k=8)
kb stats                  Show index stats + capabilities
kb reset                  Drop DB and start fresh
```

## Configuration

### Global mode (default)

`kb init` creates `~/.config/kb/config.toml`. Database lives at `~/.local/share/kb/kb.db`. Sources are absolute paths, managed with `kb add` / `kb remove`.

### Project mode

`kb init --project` creates `.kb.toml` in the current directory (found by walking up from cwd, like `.gitignore`). Database and sources are relative to the config file. Project config takes precedence over global when both exist.

### Config format

```toml
# Sources (absolute paths in global mode, relative in project mode)
sources = [
    "/home/user/notes",
    "/home/user/docs",
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

### .kbignore

Drop a `.kbignore` in any source directory. Gitignore-style syntax:

```
# Skip directories
internal/
drafts/

# Skip file patterns
*.draft.md
WIP-*
```

### secrets.toml

Optionally store secrets in `~/.config/kb/secrets.toml` instead of environment variables:

```toml
openai_api_key = "sk-..."
# For Ollama / other providers:
# openai_base_url = "http://localhost:11434/v1"
# openai_api_key = "unused"
```

Keys are loaded as uppercase environment variables. Existing env vars take precedence.

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

- **Hybrid search** — vector similarity + FTS5 keyword search, fused with Reciprocal Rank Fusion
- **Heading-aware chunking** — markdown split by heading hierarchy, each chunk carries ancestry
- **Incremental indexing** — content-hash per chunk, only re-embeds changes
- **LLM rerank** — `ask` over-fetches candidates, LLM ranks by relevance, keeps the best
- **Pre-search filters** — file globs, date ranges, keyword inclusion/exclusion
- **PDF support** — install with `kb[pdf]` or `kb[all]`
- **Pluggable chunking** — uses [chonkie](https://github.com/bhavnicksm/chonkie) when available, regex fallback otherwise

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

## Contributing

Contributions welcome! Please open an issue first to discuss what you'd like to change.

See [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) for setup, architecture, and workflow.

## License

[MIT](LICENSE)
