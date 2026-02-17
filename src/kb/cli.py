"""CLI entry point for kb."""

import json
import re
import sys
from pathlib import Path

from .api import (
    FileNotIndexedError,
    NoIndexError,
    NoSearchTermsError,
    _resolve_doc_path,
    ask_core,
    fts_core,
    list_core,
    search_core,
    similar_core,
    stats_core,
)
from .chunk import CHONKIE_AVAILABLE
from .config import (
    GLOBAL_CONFIG_DIR,
    GLOBAL_CONFIG_FILE,
    GLOBAL_CONFIG_TEMPLATE,
    GLOBAL_DATA_DIR,
    PROJECT_CONFIG_FILE,
    PROJECT_CONFIG_TEMPLATE,
    Config,
    find_config,
    load_secrets,
    save_config,
)
from .db import connect, reset
from .extract import supported_extensions, unavailable_formats
from .ingest import index_directory

USAGE = """\
kb — CLI knowledge base powered by sqlite-vec

Indexes 30+ document formats: markdown, PDF, DOCX, EPUB, HTML, ODT, RTF,
plain text, email, subtitles, and more. Optional: code files (index_code = true).

Usage:
  kb init                        Create global config (~/.config/kb/)
  kb init --project              Create project-local .kb.toml in current directory
  kb add <dir> [dir...]          Add source directories
  kb remove <dir> [dir...]       Remove source directories
  kb sources                     List configured sources
  kb index [DIR...] [--no-size-limit]  Index sources (skip files > max_file_size_mb)
  kb allow <file>                Whitelist a large file for indexing
  kb search "query" [k] [--threshold N] [--json]  Hybrid semantic + keyword search (default k=5)
  kb fts "query" [k] [--json]          Keyword-only search (no embedding, instant)
  kb ask "question" [k] [--threshold N] [--json]   RAG: search + rerank + answer (default k=8)
  kb similar <file> [k]          Find similar documents (no API call, default k=10)
  kb tag <file> tag1 [tag2...]   Add tags to a document
  kb untag <file> tag1 [tag2...]  Remove tags from a document
  kb tags                        List all tags with document counts
  kb list [--full]                List indexed documents (summary; --full for details)
  kb stats                       Show index statistics and supported formats
  kb reset                       Drop database and start fresh
  kb version                      Show version (also: kb v, kb --version)
  kb mcp                         Start MCP server (for Claude Desktop / AI agents)
  kb completion <shell>           Output shell completions (zsh, bash, fish)

Search filters (inline with query):
  file:articles/*.md             Glob filter on file path
  type:markdown                  Filter by document type (markdown, pdf, etc.)
  tag:python                     Filter by tag
  dt>"2026-02-01"                After date
  dt<"2026-02-14"                Before date
  +"keyword"                     Must contain
  -"keyword"                     Must not contain

Examples:
  kb init                        # global mode (default)
  kb add ~/notes ~/docs          # add sources
  kb index                       # index all sources
  kb search 'file:articles/*.md cost optimization'
  kb search 'type:pdf tag:python machine learning'
  kb ask 'dt>"2026-02-01" what deployment patterns?'
  kb similar docs/guide.md       # find related documents
  kb tag docs/guide.md python tutorial  # add tags
  kb init --project              # project-local mode
"""


def cmd_init(project: bool):
    if project:
        cfg_path = Path.cwd() / PROJECT_CONFIG_FILE
        if cfg_path.exists():
            print(f"{PROJECT_CONFIG_FILE} already exists at {cfg_path}")
            sys.exit(1)
        cfg_path.write_text(PROJECT_CONFIG_TEMPLATE)
        print(f"Created {cfg_path}")
        print("Edit 'sources' to add directories to index, then run: kb index")
    else:
        if GLOBAL_CONFIG_FILE.exists():
            print(f"Global config already exists at {GLOBAL_CONFIG_FILE}")
            sys.exit(1)
        GLOBAL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        GLOBAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
        GLOBAL_CONFIG_FILE.write_text(GLOBAL_CONFIG_TEMPLATE)
        print(f"Created {GLOBAL_CONFIG_FILE}")
        print(f"Database: {GLOBAL_DATA_DIR / 'kb.db'}")
        print("Add sources with: kb add ~/notes ~/docs")


def cmd_add(cfg: Config, dirs: list[str]):
    if not dirs:
        print("Usage: kb add <dir> [dir...]")
        sys.exit(1)

    for d in dirs:
        p = Path(d).expanduser().resolve()
        if not p.is_dir():
            print(f"Not a directory: {d}")
            sys.exit(1)

        if cfg.scope == "global":
            entry = str(p)
        else:
            try:
                entry = str(p.relative_to(cfg.config_dir))
            except ValueError:
                entry = str(p)

        if entry in cfg.sources:
            print(f"  Already added: {entry}")
            continue

        cfg.sources.append(entry)
        print(f"  Added: {entry}")

    save_config(cfg)
    print(f"Saved {cfg.config_path}")


def cmd_remove(cfg: Config, dirs: list[str]):
    if not dirs:
        print("Usage: kb remove <dir> [dir...]")
        sys.exit(1)

    for d in dirs:
        p = Path(d).expanduser().resolve()
        if cfg.scope == "global":
            entry = str(p)
        else:
            try:
                entry = str(p.relative_to(cfg.config_dir))
            except ValueError:
                entry = str(p)

        if entry in cfg.sources:
            cfg.sources.remove(entry)
            print(f"  Removed: {entry}")
        else:
            print(f"  Not found: {entry}")

    save_config(cfg)
    print(f"Saved {cfg.config_path}")


def cmd_sources(cfg: Config):
    if not cfg.sources:
        print("No sources configured. Run: kb add <dir>")
        return
    for s in cfg.sources:
        p = Path(s).expanduser() if cfg.scope == "global" else cfg.config_dir / s
        exists = p.is_dir()
        marker = " " if exists else " (missing)"
        print(f"  {s}{marker}")


def cmd_allow(cfg: Config, files: list[str]):
    if not files:
        print("Usage: kb allow <file> [file...]")
        sys.exit(1)
    if not cfg.config_path:
        print("No config found. Run 'kb init' first.")
        sys.exit(1)

    for f in files:
        p = Path(f).expanduser().resolve()
        if not p.is_file():
            print(f"Not a file: {f}")
            sys.exit(1)

        if cfg.scope == "global":
            entry = str(p)
        else:
            try:
                entry = str(p.relative_to(cfg.config_dir))
            except ValueError:
                entry = str(p)

        if entry in cfg.allowed_large_files:
            print(f"  Already allowed: {entry}")
            continue

        cfg.allowed_large_files.append(entry)
        print(f"  Allowed: {entry}")

    save_config(cfg)
    print(f"Saved {cfg.config_path}")


def cmd_index(cfg: Config, args: list[str]):
    no_size_limit = "--no-size-limit" in args
    dir_args = [a for a in args if a != "--no-size-limit"]

    if dir_args:
        dirs = [Path(a).resolve() for a in dir_args]
    elif cfg.source_paths:
        dirs = cfg.source_paths
    else:
        print("No sources configured. Either:")
        print("  1. Run 'kb add <dir>' to add source directories")
        print("  2. Pass directories explicitly: kb index ~/docs ~/notes")
        sys.exit(1)

    for dir_path in dirs:
        if not dir_path.is_dir():
            print(f"Not a directory: {dir_path}")
            sys.exit(1)
        index_directory(dir_path, cfg, no_size_limit=no_size_limit)


def _best_snippet(text: str, query: str, width: int = 500) -> str:
    """Return a snippet of text centered around query term matches."""
    if not text or len(text) <= width:
        return text or ""
    words = re.findall(r"\w+", query.lower())
    if not words:
        return text[:width]
    lower = text.lower()
    positions = []
    for w in words:
        idx = lower.find(w)
        if idx >= 0:
            positions.append(idx)
    if not positions:
        return text[:width]
    center = sum(positions) // len(positions)
    start = max(0, center - width // 2)
    end = min(len(text), start + width)
    start = max(0, end - width)
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(text) else ""
    return prefix + text[start:end] + suffix


def cmd_search(
    query: str,
    cfg: Config,
    top_k: int = 5,
    threshold: float | None = None,
    json_output: bool = False,
):
    try:
        result = search_core(query, cfg, top_k, threshold)
    except NoIndexError as e:
        print(str(e))
        sys.exit(1)

    if json_output:
        print(json.dumps(result, ensure_ascii=False))
        return

    clean_query = result["query"]
    timing = result["timing_ms"]
    candidates = result["candidates"]

    if result.get("filters"):
        print(f"Filters: {', '.join(f'{k}={v}' for k, v in result['filters'].items())}")

    print(f'Query: "{clean_query}"')
    print(
        f"Embed: {timing['embed']}ms | Vec: {timing['vec']}ms | FTS: {timing['fts']}ms"
    )
    print(
        f"Candidates: {candidates['vec']} vec, {candidates['fts']} fts -> {candidates['fused']} fused\n"
    )

    for r in result["results"]:
        sim = (
            f"sim:{r['similarity']:.3f}" if r["similarity"] is not None else "fts-only"
        )
        source_tag = "+".join(r["sources"])
        print(
            f"--- [{r['rank']}] {r['doc_path']} ({sim}, {source_tag}, rrf:{r['rrf_score']:.4f}) ---"
        )
        if r["heading"]:
            print(f"    Section: {r['heading']}")
        preview = _best_snippet(r["text"] or "", clean_query).replace("\n", "\n    ")
        print(f"    {preview}")
        if r["text"] and len(r["text"]) > 500:
            print(f"    ({len(r['text'])} chars total)")
        print()


def cmd_fts(
    query: str,
    cfg: Config,
    top_k: int = 5,
    json_output: bool = False,
):
    """FTS-only keyword search — no embedding, no API cost."""
    try:
        result = fts_core(query, cfg, top_k)
    except NoIndexError as e:
        print(str(e))
        sys.exit(1)
    except NoSearchTermsError as e:
        print(str(e))
        sys.exit(1)

    if json_output:
        print(json.dumps(result, ensure_ascii=False))
        return

    clean_query = result["query"]
    fts_ms = result["timing_ms"]["fts"]

    if result.get("filters"):
        print(f"Filters: {', '.join(f'{k}={v}' for k, v in result['filters'].items())}")

    print(f'Query: "{clean_query}"')
    print(f"FTS: {fts_ms}ms | {len(result['results'])} results\n")

    for r in result["results"]:
        bm25_str = f"bm25:{r['bm25']:.3f}"
        print(f"--- [{r['rank']}] {r['doc_path']} ({bm25_str}) ---")
        if r["heading"]:
            print(f"    Section: {r['heading']}")
        preview = _best_snippet(r["text"] or "", clean_query).replace("\n", "\n    ")
        print(f"    {preview}")
        if r["text"] and len(r["text"]) > 500:
            print(f"    ({len(r['text'])} chars total)")
        print()


def cmd_ask(
    question: str,
    cfg: Config,
    top_k: int = 8,
    threshold: float | None = None,
    json_output: bool = False,
):
    """Full RAG: hybrid retrieve -> filter -> LLM rerank -> confidence filter -> answer."""
    try:
        result = ask_core(question, cfg, top_k, threshold)
    except NoIndexError as e:
        print(str(e))
        sys.exit(1)

    if json_output:
        out = {
            "question": result["question"],
            "answer": result["answer"],
            "model": result["model"],
            "bm25_shortcut": result["bm25_shortcut"],
            "timing_ms": result["timing_ms"],
            "tokens": result["tokens"],
            "sources": result["sources"],
        }
        print(json.dumps(out, ensure_ascii=False))
        return

    clean_question = result["question"]
    timing = result["timing_ms"]
    bm25_shortcut = result["bm25_shortcut"]
    rerank_info = result.get("rerank")

    if result.get("filters"):
        print(f"Filters: {', '.join(f'{k}={v}' for k, v in result['filters'].items())}")

    shortcut_tag = " (bm25 shortcut)" if bm25_shortcut else ""
    print(f"Q: {clean_question}")
    print(
        f"(embed: {timing['embed']}ms | search: {timing['search']}ms | generate: {timing['generate']}ms{shortcut_tag})"
    )

    if result["answer"] is None:
        print("\nNo relevant documents found.")
        return

    if rerank_info:
        print(
            f"(rerank: {rerank_info['rerank_ms']:.0f}ms, "
            f"{rerank_info['prompt_tokens']}+{rerank_info['completion_tokens']} tokens, "
            f"{rerank_info['input_count']} -> {rerank_info['output_count']})"
        )

    print(f"(model: {result['model']})")
    print(
        f"(tokens: {result['tokens']['prompt']} in / {result['tokens']['completion']} out)"
    )
    print(
        f"(results: {result.get('result_count', '?')} retrieved, "
        f"{result.get('filtered_count', '?')} above threshold)\n"
    )
    print(result["answer"])
    print("\n--- Sources ---")
    for src in result["sources"]:
        if src["heading"]:
            print(f"  [{src['rank']}] {src['doc_path']} > {src['heading']}")
        else:
            print(f"  [{src['rank']}] {src['doc_path']}")


def cmd_similar(file_arg: str, cfg: Config, top_k: int = 10):
    try:
        result = similar_core(file_arg, cfg, top_k)
    except NoIndexError as e:
        print(str(e))
        sys.exit(1)
    except FileNotIndexedError as e:
        print(str(e))
        if "not in index" in str(e).lower():
            print("Run 'kb index' to index it first.")
        sys.exit(1)

    if not result["results"]:
        print(f"No similar documents found for {result['source']}.")
        return

    print(f"Documents similar to: {result['source']}\n")
    for r in result["results"]:
        print(f"--- [{r['rank']}] {r['doc_path']} (sim:{r['similarity']:.3f}) ---")
        if r["title"]:
            print(f"    {r['title']}")


def cmd_tag(cfg: Config, file_arg: str, new_tags: list[str]):
    if not cfg.db_path.exists():
        print("No index found. Run 'kb index' first.")
        sys.exit(1)

    conn = connect(cfg)
    doc_path = _resolve_doc_path(cfg, conn, file_arg)
    if not doc_path:
        print(f"File not in index: {file_arg}")
        conn.close()
        sys.exit(1)

    row = conn.execute(
        "SELECT tags FROM documents WHERE path = ?", (doc_path,)
    ).fetchone()
    existing = {t.strip().lower() for t in (row["tags"] or "").split(",") if t.strip()}
    existing.update(t.lower() for t in new_tags)
    conn.execute(
        "UPDATE documents SET tags = ? WHERE path = ?",
        (",".join(sorted(existing)), doc_path),
    )
    conn.commit()
    print(f"Tags for {doc_path}: {', '.join(sorted(existing))}")
    conn.close()


def cmd_untag(cfg: Config, file_arg: str, remove_tags: list[str]):
    if not cfg.db_path.exists():
        print("No index found. Run 'kb index' first.")
        sys.exit(1)

    conn = connect(cfg)
    doc_path = _resolve_doc_path(cfg, conn, file_arg)
    if not doc_path:
        print(f"File not in index: {file_arg}")
        conn.close()
        sys.exit(1)

    row = conn.execute(
        "SELECT tags FROM documents WHERE path = ?", (doc_path,)
    ).fetchone()
    existing = {t.strip().lower() for t in (row["tags"] or "").split(",") if t.strip()}
    existing -= {t.lower() for t in remove_tags}
    conn.execute(
        "UPDATE documents SET tags = ? WHERE path = ?",
        (",".join(sorted(existing)), doc_path),
    )
    conn.commit()
    if existing:
        print(f"Tags for {doc_path}: {', '.join(sorted(existing))}")
    else:
        print(f"All tags removed from {doc_path}")
    conn.close()


def cmd_tags(cfg: Config):
    if not cfg.db_path.exists():
        print("No index found. Run 'kb index' first.")
        sys.exit(1)

    conn = connect(cfg)
    rows = conn.execute("SELECT tags FROM documents WHERE tags != ''").fetchall()
    conn.close()

    if not rows:
        print("No tagged documents.")
        return

    counts: dict[str, int] = {}
    for r in rows:
        for tag in r["tags"].split(","):
            tag = tag.strip().lower()
            if tag:
                counts[tag] = counts.get(tag, 0) + 1

    print(f"{len(counts)} tags across {len(rows)} documents\n")
    for tag, count in sorted(counts.items()):
        print(f"  {tag:<30} {count} doc{'s' if count != 1 else ''}")


def cmd_stats(cfg: Config):
    result = stats_core(cfg)
    if "error" in result:
        print(result["error"])
        return

    db_size_kb = result["db_size_bytes"] / 1024
    print(f"DB: {result['db_path']} ({db_size_kb:.1f} KB)")
    print(f"Documents: {result['doc_count']}", end="")
    if result["type_counts"]:
        parts = [f"{cnt} {t}" for t, cnt in result["type_counts"].items()]
        print(f" ({', '.join(parts)})", end="")
    print()
    print(
        f"Chunks: {result['chunk_count']} | Vectors: {result['vec_count']} "
        f"| FTS entries: {result['fts_count']}"
    )
    print(
        f"Total text: {result['total_chars']:,} chars "
        f"(~{result['total_chars'] // 4:,} tokens)"
    )

    print("\nCapabilities:")
    print(
        f"  chonkie chunking:   "
        f"{'yes' if CHONKIE_AVAILABLE else 'no (pip install chonkie)'}"
    )
    print(
        f"  LLM rerank:         yes (ask mode, "
        f"top-{cfg.rerank_fetch_k} -> top-{cfg.rerank_top_k})"
    )
    print('  Pre-search filters: yes (file:, type:, tag:, dt>, dt<, +"kw", -"kw")')
    print(
        f"  Index code files:   "
        f"{'yes' if cfg.index_code else 'no (set index_code = true)'}"
    )

    exts = sorted(supported_extensions(include_code=cfg.index_code))
    print(f"  Supported formats:  {', '.join(exts)}")

    missing = unavailable_formats()
    if missing:
        for ext, pkg in missing:
            print(f"  {ext}: unavailable (pip install {pkg})")

    print("\nDocuments:")
    for doc in result["documents"]:
        h = doc["content_hash"][:8] if doc["content_hash"] else "n/a"
        type_tag = f" [{doc['type']}]" if doc["type"] != "markdown" else ""
        print(
            f"  {doc['path']}: {doc['chunk_count']} chunks [{h}]{type_tag} ({doc['title']})"
        )


def _format_size(size: int) -> str:
    if size >= 1_000_000:
        return f"{size / 1_000_000:.1f} MB"
    if size >= 1_000:
        return f"{size / 1_000:.1f} KB"
    return f"{size} B"


def cmd_list(cfg: Config, full: bool = False):
    result = list_core(cfg)
    if "error" in result:
        print(result["error"])
        return

    rows = result["documents"]
    if not rows:
        print("No documents indexed.")
        return

    if full:
        print(f"{len(rows)} documents indexed\n")
        for r in rows:
            path = r["path"]
            doc_type = r["type"] or "unknown"
            chunks = r["chunk_count"]
            size = r["size_bytes"]
            date = (r["indexed_at"] or "")[:10]
            print(
                f"  {path:<50} {doc_type:<12} {chunks:>3} chunks  "
                f"{_format_size(size):>10}  {date}"
            )
        return

    type_stats: dict[str, dict] = {}
    total_size = 0
    total_chunks = 0
    for r in rows:
        doc_type = r["type"] or "unknown"
        size = r["size_bytes"]
        chunks = r["chunk_count"]
        total_size += size
        total_chunks += chunks
        if doc_type not in type_stats:
            type_stats[doc_type] = {"count": 0, "size": 0, "chunks": 0}
        type_stats[doc_type]["count"] += 1
        type_stats[doc_type]["size"] += size
        type_stats[doc_type]["chunks"] += chunks

    print(
        f"{len(rows)} documents indexed "
        f"({_format_size(total_size)}, {total_chunks} chunks)\n"
    )
    for doc_type in sorted(
        type_stats, key=lambda t: type_stats[t]["count"], reverse=True
    ):
        s = type_stats[doc_type]
        print(
            f"  {doc_type:<12} {s['count']:>4} docs  {s['chunks']:>5} chunks  "
            f"{_format_size(s['size']):>10}"
        )
    print("\nUse 'kb list --full' for per-file details.")


def cmd_completion(shell: str):
    subcommands = (
        "init add remove sources index allow search fts ask similar "
        "tag untag tags stats reset list version mcp completion"
    )

    if shell == "zsh":
        print(
            f"""\
_kb() {{
  local -a commands
  commands=({subcommands})
  _arguments '1:command:({" ".join(subcommands.split())})' '*:file:_files'
}}
compdef _kb kb"""
        )
    elif shell == "bash":
        print(
            f"""\
_kb() {{
  local cur commands
  COMPREPLY=()
  cur="${{COMP_WORDS[COMP_CWORD]}}"
  if [[ $COMP_CWORD -eq 1 ]]; then
    commands="{subcommands}"
    COMPREPLY=( $(compgen -W "$commands" -- "$cur") )
  else
    case "${{COMP_WORDS[1]}}" in
      add|remove|index|allow)
        COMPREPLY=( $(compgen -d -- "$cur") )
        ;;
      init)
        COMPREPLY=( $(compgen -W "--project" -- "$cur") )
        ;;
      search|ask)
        COMPREPLY=( $(compgen -W "--threshold --json" -- "$cur") )
        ;;
      fts)
        COMPREPLY=( $(compgen -W "--json" -- "$cur") )
        ;;
      completion)
        COMPREPLY=( $(compgen -W "zsh bash fish" -- "$cur") )
        ;;
    esac
  fi
}}
complete -F _kb kb"""
        )
    elif shell == "fish":
        cmds = subcommands.split()
        print("# Fish completions for kb")
        for c in cmds:
            print(f"complete -c kb -n '__fish_use_subcommand' -a {c}")
        print(
            "complete -c kb -n '__fish_seen_subcommand_from add remove index allow' -F"
        )
        print("complete -c kb -n '__fish_seen_subcommand_from init' -a '--project'")
        print(
            "complete -c kb -n '__fish_seen_subcommand_from search ask' "
            "-a '--threshold --json'"
        )
        print("complete -c kb -n '__fish_seen_subcommand_from fts' -a '--json'")
        print(
            "complete -c kb -n '__fish_seen_subcommand_from completion' "
            "-a 'zsh bash fish'"
        )
    else:
        print(f"Unsupported shell: {shell}")
        print("Supported: zsh, bash, fish")
        sys.exit(1)


def main():
    load_secrets()
    args = sys.argv[1:]

    if not args:
        print(USAGE)
        sys.exit(1)

    cmd = args[0]

    if cmd in ("-h", "--help", "help"):
        print(USAGE)
        sys.exit(0)

    if cmd in ("version", "v", "--version"):
        from importlib.metadata import version

        print(f"kb {version('kb')}")
        sys.exit(0)

    if cmd == "init":
        if len(args) > 1 and args[1] in ("-h", "--help"):
            print("Usage: kb init [--project]")
            sys.exit(0)
        project = "--project" in args[1:]
        cmd_init(project)
        sys.exit(0)

    if cmd == "completion":
        if len(args) < 2 or args[1] in ("-h", "--help"):
            print("Usage: kb completion <zsh|bash|fish>")
            sys.exit(0 if len(args) > 1 and args[1] in ("-h", "--help") else 1)
        cmd_completion(args[1])
        sys.exit(0)

    if cmd == "mcp":
        from .mcp_server import main as mcp_main

        mcp_main()
        sys.exit(0)

    # All other commands need config
    cfg = find_config()

    scope_label = f"[{cfg.scope}]" if cfg.config_path else "[no config]"
    if cfg.config_path:
        print(f"Config: {cfg.config_path} {scope_label}")

    # Per-subcommand help
    sub_help = len(args) > 1 and args[1] in ("-h", "--help")

    if cmd == "add":
        if not cfg.config_path:
            print("No config found. Run 'kb init' first.")
            sys.exit(1)
        if sub_help or not args[1:]:
            print("Usage: kb add <dir> [dir...]")
            sys.exit(0)
        cmd_add(cfg, args[1:])
    elif cmd == "allow":
        if sub_help or not args[1:]:
            print("Usage: kb allow <file>")
            sys.exit(0)
        cmd_allow(cfg, args[1:])
    elif cmd == "remove":
        if not cfg.config_path:
            print("No config found. Run 'kb init' first.")
            sys.exit(1)
        if sub_help or not args[1:]:
            print("Usage: kb remove <dir> [dir...]")
            sys.exit(0)
        cmd_remove(cfg, args[1:])
    elif cmd == "sources":
        if sub_help:
            print("Usage: kb sources")
            sys.exit(0)
        cmd_sources(cfg)
    elif cmd == "index":
        if sub_help:
            print("Usage: kb index [DIR...] [--no-size-limit]")
            sys.exit(0)
        cmd_index(cfg, args[1:])
    elif cmd == "search":
        if len(args) < 2 or sub_help:
            print('Usage: kb search "query" [k] [--threshold N] [--json]')
            sys.exit(0 if sub_help else 1)
        threshold = None
        search_args = list(args[1:])
        json_out = "--json" in search_args
        if json_out:
            search_args.remove("--json")
        if "--threshold" in search_args:
            ti = search_args.index("--threshold")
            if ti + 1 < len(search_args):
                threshold = float(search_args[ti + 1])
                del search_args[ti : ti + 2]
            else:
                print("--threshold requires a value")
                sys.exit(1)
        top_k = int(search_args[1]) if len(search_args) > 1 else 5
        cmd_search(
            search_args[0], cfg, top_k, threshold=threshold, json_output=json_out
        )
    elif cmd == "fts":
        if len(args) < 2 or sub_help:
            print('Usage: kb fts "query" [k] [--json]')
            sys.exit(0 if sub_help else 1)
        fts_args = list(args[1:])
        json_out = "--json" in fts_args
        if json_out:
            fts_args.remove("--json")
        top_k = int(fts_args[1]) if len(fts_args) > 1 else 5
        cmd_fts(fts_args[0], cfg, top_k, json_output=json_out)
    elif cmd == "ask":
        if len(args) < 2 or sub_help:
            print('Usage: kb ask "question" [k] [--threshold N] [--json]')
            sys.exit(0 if sub_help else 1)
        threshold = None
        ask_args = list(args[1:])
        json_out = "--json" in ask_args
        if json_out:
            ask_args.remove("--json")
        if "--threshold" in ask_args:
            ti = ask_args.index("--threshold")
            if ti + 1 < len(ask_args):
                threshold = float(ask_args[ti + 1])
                del ask_args[ti : ti + 2]
            else:
                print("--threshold requires a value")
                sys.exit(1)
        question = ask_args[0]
        top_k = int(ask_args[1]) if len(ask_args) > 1 else 8
        cmd_ask(question, cfg, top_k, threshold=threshold, json_output=json_out)
    elif cmd == "similar":
        if len(args) < 2 or sub_help:
            print("Usage: kb similar <file> [k]")
            sys.exit(0 if sub_help else 1)
        top_k = int(args[2]) if len(args) > 2 else 10
        cmd_similar(args[1], cfg, top_k)
    elif cmd == "tag":
        if len(args) < 3 or sub_help:
            print("Usage: kb tag <file> tag1 [tag2...]")
            sys.exit(0 if sub_help else 1)
        cmd_tag(cfg, args[1], args[2:])
    elif cmd == "untag":
        if len(args) < 3 or sub_help:
            print("Usage: kb untag <file> tag1 [tag2...]")
            sys.exit(0 if sub_help else 1)
        cmd_untag(cfg, args[1], args[2:])
    elif cmd == "tags":
        if sub_help:
            print("Usage: kb tags")
            sys.exit(0)
        cmd_tags(cfg)
    elif cmd == "list":
        if sub_help:
            print("Usage: kb list [--full]")
            sys.exit(0)
        cmd_list(cfg, full="--full" in args)
    elif cmd == "stats":
        if sub_help:
            print("Usage: kb stats")
            sys.exit(0)
        cmd_stats(cfg)
    elif cmd == "reset":
        if sub_help:
            print("Usage: kb reset")
            sys.exit(0)
        reset(cfg.db_path)
    else:
        print(f"Unknown command: {cmd}")
        print(USAGE)
        sys.exit(1)
