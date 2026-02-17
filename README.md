# kb (knowledge base)

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

CLI RAG tool for your docs. Index 30+ document formats (markdown, PDF, DOCX, EPUB, HTML, ODT, RTF, plain text, email, and more), hybrid search (semantic + keyword), ask questions and get sourced answers. Built on [sqlite-vec](https://github.com/asg017/sqlite-vec).

## Features

- **Hybrid search** — vector similarity + FTS5 keyword search, fused with Reciprocal Rank Fusion (with rank bonuses)
- **Keyword-only search** — `kb fts` for instant BM25 results with zero API cost
- **Heading-aware chunking** — markdown split by heading hierarchy, each chunk carries ancestry
- **Incremental indexing** — content-hash per chunk, only re-embeds changes
- **LLM rerank** — `ask` over-fetches candidates, LLM ranks by relevance, keeps the best
- **Pre-search filters** — file globs, document type, tags, date ranges, keyword inclusion/exclusion
- **Document tagging** — manual tags via `kb tag`, auto-parsed from markdown frontmatter
- **Similar documents** — find related docs using stored embeddings (no API call)
- **30+ formats** — markdown, PDF, DOCX, PPTX, XLSX, EPUB, HTML, ODT, ODS, ODP, RTF, email (.eml), subtitles (.srt/.vtt), and plain text variants (.txt, .rst, .org, .csv, .json, .yaml, .tex, etc.)
- **Optional code indexing** — set `index_code = true` to also index source code files (.py, .js, .ts, .go, .rs, etc.)
- **Pluggable chunking** — uses [chonkie](https://github.com/bhavnicksm/chonkie) when available, regex fallback otherwise
- **MCP server** — expose kb as tools for Claude Desktop, Claude Code, and other MCP clients

## Install

```bash
# One-liner (installs uv if needed)
curl -LsSf https://github.com/ariel-frischer/kb/raw/main/install.sh | sh

# Or with uv directly (all optional deps: PDF, Office, RTF, chunking)
uv tool install --from "git+https://github.com/ariel-frischer/kb.git" "kb[all]"

# Minimal (markdown, HTML, plain text, email, EPUB, ODT — no extra deps)
uv tool install --from "git+https://github.com/ariel-frischer/kb.git" kb

# Pick extras individually
uv tool install --from "git+https://github.com/ariel-frischer/kb.git" "kb[pdf]"       # + PDF
uv tool install --from "git+https://github.com/ariel-frischer/kb.git" "kb[office]"    # + DOCX, PPTX, XLSX
uv tool install --from "git+https://github.com/ariel-frischer/kb.git" "kb[rtf]"       # + RTF
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

# 4. Search (hybrid: semantic + keyword)
kb search "deployment patterns"

# 5. Quick keyword search (instant, no API cost)
kb fts "deployment patterns"

# 6. Ask (RAG: search → rerank → answer)
kb ask "what are the recommended deployment patterns?"

# 7. List indexed documents
kb list

# 8. Check what's indexed
kb stats
```

## Commands

```
kb init                        Create global config (~/.config/kb/)
kb init --project              Create project-local .kb.toml in current directory
kb add <dir> [dir...]          Add source directories
kb remove <dir> [dir...]       Remove source directories
kb sources                     List configured sources
kb index [DIR...]              Index sources from config (or explicit dirs)
kb search "query" [k] [--threshold N] [--json]  Hybrid search (default k=5)
kb fts "query" [k] [--json]            Keyword-only search (instant, no API cost)
kb ask "question" [k] [--threshold N] [--json]  RAG answer (default k=8, BM25 shortcut when confident)
kb list                        Summary of indexed documents by type
kb list --full                 List every indexed document with metadata
kb similar <file> [k]          Find similar documents (no API call, default k=10)
kb tag <file> tag1 [tag2...]   Add tags to a document
kb untag <file> tag1 [tag2...]  Remove tags from a document
kb tags                        List all tags with document counts
kb stats                       Show index stats + capabilities
kb reset                       Drop DB and start fresh
kb version                     Show version (also: kb v, kb --version)
kb mcp                         Start MCP server (for Claude Desktop / AI agents)
kb completion <shell>          Output shell completions (zsh, bash, fish)
```

### Shell completions

```bash
# Zsh (add to ~/.zshrc)
eval "$(kb completion zsh)"

# Bash (add to ~/.bashrc)
eval "$(kb completion bash)"

# Fish (add to ~/.config/fish/config.fish)
kb completion fish | source
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
# search_threshold = 0.001      # min cosine similarity for `kb search` (also --threshold flag)
# ask_threshold = 0.001         # min cosine similarity for `kb ask` (also --threshold flag)
# rerank_fetch_k = 20
# rerank_top_k = 5
# index_code = false       # set true to also index source code files
```

### .kbignore

Drop a `.kbignore` in any source directory to exclude files from indexing. Uses fnmatch glob syntax with `#` comments.

Lookup order: checks `<source-dir>/.kbignore`, then `<source-dir>/../.kbignore` (first found wins).

```
# Skip directories (trailing slash)
drafts/
.obsidian/
node_modules/

# Skip file patterns
*.draft.md
WIP-*
CHANGELOG.md
```

See [docs/kbignore.md](docs/kbignore.md) for common patterns by use case.

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
kb search 'type:pdf machine learning'
kb search 'tag:python tutorial basics'
kb search 'dt>"2026-02-01" recent developments'
kb search '+"docker" -"kubernetes" container setup'
kb ask 'file:briefs/*.pdf dt>"2026-02-13" what are the costs?'
```

| Filter | Syntax | Example |
|---|---|---|
| File glob | `file:<pattern>` | `file:articles/*.md` |
| Document type | `type:<type>` | `type:markdown`, `type:pdf` |
| Tag | `tag:<name>` | `tag:python` |
| After date | `dt>"YYYY-MM-DD"` | `dt>"2026-02-01"` |
| Before date | `dt<"YYYY-MM-DD"` | `dt<"2026-02-14"` |
| Must contain | `+"keyword"` | `+"docker"` |
| Must not contain | `-"keyword"` | `-"kubernetes"` |

## Tags

Tag documents manually or let `kb index` auto-parse tags from markdown frontmatter:

```yaml
---
tags: [python, tutorial]
---
```

```bash
kb tag docs/guide.md python tutorial   # add tags manually
kb untag docs/guide.md tutorial        # remove a tag
kb tags                                # list all tags with counts
kb search 'tag:python basics'          # filter by tag in search
```

## Supported Formats

**Always available (no extra deps):**

| Category | Extensions |
|----------|-----------|
| Markdown | `.md`, `.markdown` |
| Plain text | `.txt`, `.text`, `.rst`, `.org`, `.log`, `.csv`, `.tsv`, `.json`, `.yaml`, `.yml`, `.toml`, `.xml`, `.ini`, `.cfg`, `.tex`, `.latex`, `.bib`, `.nfo`, `.adoc`, `.asciidoc`, `.properties` |
| HTML | `.html`, `.htm`, `.xhtml` |
| Subtitles | `.srt`, `.vtt` |
| Email | `.eml` |
| OpenDocument | `.odt`, `.ods`, `.odp` |
| EPUB | `.epub` |

**Optional (install with extras):**

| Category | Extensions | Install |
|----------|-----------|---------|
| PDF | `.pdf` | `kb[pdf]` or `kb[all]` |
| Office | `.docx`, `.pptx`, `.xlsx` | `kb[office]` or `kb[all]` |
| RTF | `.rtf` | `kb[rtf]` or `kb[all]` |

**Code files (opt-in):** Set `index_code = true` in config to also index source code — `.py`, `.js`, `.ts`, `.go`, `.rs`, `.java`, `.c`, `.cpp`, and 60+ more extensions.

Run `kb stats` to see which formats are available in your installation.

## How It Works

```
kb index
  1. Find files matching supported formats (respecting .kbignore)
  2. Extract text (format-specific: markdown, PDF, DOCX, HTML, etc.)
  3. Content-hash check — skip unchanged files
  4. Chunk (chonkie or regex fallback)
  5. Diff chunks by hash — only embed new/changed
  6. Batch embed via OpenAI
  7. Store in sqlite-vec (vec0) + FTS5

kb search "query"
  1. Parse filters, strip from query
  2. Embed clean query
  3. Vector search (vec0 MATCH) + FTS5 keyword search
  4. Fuse with RRF (rank bonuses for top positions)
  5. Apply filters
  6. Display results

kb fts "query"
  1. Parse filters, strip from query
  2. FTS5 keyword search (no embedding)
  3. Normalize BM25 scores
  4. Apply filters
  5. Display results (instant, zero API cost)

kb ask "question"
  1. BM25 probe — if top FTS hit is high-confidence, skip to step 5
  2. Same as search, but over-fetch 20
  3. Apply filters
  4. LLM rerank -> top 5
  5. Confidence threshold
  6. LLM generates answer from context
```

## MCP Server

kb includes an [MCP](https://modelcontextprotocol.io/) server that exposes search and ask as tools for AI agents.

### Setup with Claude Desktop

Add to your Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS, `~/.config/Claude/claude_desktop_config.json` on Linux):

```json
{
  "mcpServers": {
    "kb": {
      "command": "kb-mcp"
    }
  }
}
```

### Setup with Claude Code

```bash
claude mcp add kb kb-mcp
```

### Available tools

| Tool | Description |
|------|-------------|
| `kb_search` | Hybrid semantic + keyword search with inline filters |
| `kb_ask` | Full RAG pipeline: search + rerank + LLM answer |
| `kb_fts` | Keyword-only search (no API cost) |
| `kb_similar` | Find similar documents (no API call) |
| `kb_status` | Index statistics |
| `kb_list` | List indexed documents |

The MCP server requires the `mcp` extra: `kb[mcp]` or `kb[all]`.

## Alternatives

| Tool | What it is | Local-only | CLI | Setup |
|------|-----------|:----------:|:---:|-------|
| **kb** | CLI RAG tool — hybrid search + Q&A over 30+ document formats | Yes | Yes | `uv tool install`, single SQLite file |
| [Khoj](https://github.com/khoj-ai/khoj) | Self-hosted AI second brain with web UI, mobile, Obsidian/Emacs plugins | Optional | No | Docker or pip, runs a web server |
| [Reor](https://github.com/reorproject/reor) | Desktop note-taking app with auto-linking and local LLM | Yes | No | Electron app, uses LanceDB + Ollama |
| [LlamaIndex](https://github.com/run-llama/llama_index) | Framework for building RAG pipelines | Depends | No | Python library, you build the app |
| [ChromaDB](https://github.com/chroma-core/chroma) | Vector database with simple API | Yes | No | Python library, you build the app |
| [grepai](https://github.com/yoanbernabeu/grepai) | Semantic code search + call graphs, 100% local | Yes | Yes | `brew install` or curl, uses Ollama/OpenAI embeddings |

**When to use what:**

- **kb** — you want a CLI RAG tool that indexes docs (markdown, PDFs, DOCX, EPUB, HTML, and more) and answers questions from them
- **grepai** — you want semantic search over code (find by intent, trace call graphs), no RAG
- **Khoj** — you want a full-featured app with web UI, phone access, Obsidian integration, and agent capabilities
- **Reor** — you want an Obsidian-like desktop editor that auto-links notes using local AI
- **LlamaIndex / ChromaDB** — you're building your own RAG pipeline and need libraries, not a finished tool

## Contributing

Contributions welcome! Please open an issue first to discuss what you'd like to change.

See [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) for setup, architecture, and workflow.

## Maintenance

This is a personal tool I've open-sourced. I may or may not respond to issues/PRs. Fork freely.

## License

[MIT](LICENSE)
