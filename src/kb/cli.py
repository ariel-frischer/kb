"""CLI entry point for kb."""

import csv
import io
import json
import re
import sys
from copy import copy
from pathlib import Path

from .api import (
    FileNotIndexedError,
    KBError,
    NoIndexError,
    NoSearchTermsError,
    _resolve_doc_path,
    ask_core,
    feedback_core,
    fts_core,
    list_core,
    list_feedback_core,
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
    _project_db_path,
    find_config,
    load_secrets,
    save_config,
)
from .cost import format_usd
from .db import connect, reset
from .extract import supported_extensions, unavailable_formats
from .ingest import index_directory
from .terminal import label, print_error, result_header, style

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
  kb search "query" [k] [--threshold N] [--expand] [--json|--csv|--md]  Hybrid search (default k=5)
  kb fts "query" [k] [--json|--csv|--md]          Keyword-only search (no embedding, instant)
  kb ask "question" [k] [--threshold N] [--expand] [--json|--csv|--md]  RAG: search + answer (default k=8)
  kb similar <file> [k]          Find similar documents (no API call, default k=10)
  kb tag <file> tag1 [tag2...]   Add tags to a document
  kb untag <file> tag1 [tag2...]  Remove tags from a document
  kb tags                        List all tags with document counts
  kb list [--full]                List indexed documents (summary; --full for details)
  kb stats                       Show index statistics and supported formats
  kb reset                       Drop database and start fresh
  kb version                      Show version (also: kb v, kb --version)
  kb feedback "msg" [--tool T] [--severity bug|suggestion|note] [--context C] [--agent-id A] [--error-trace E]
                                 Submit feedback (for agents / dev use)
  kb feedback --list             List all feedback entries
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
            print_error(f"{PROJECT_CONFIG_FILE} already exists at {cfg_path}")
            sys.exit(1)
        cfg_path.write_text(PROJECT_CONFIG_TEMPLATE)
        print(style(f"Created {cfg_path}", "success"))
        db_path = _project_db_path(Path.cwd())
        print(label("Database", style(db_path, "path")))
        print(
            style(
                "Edit 'sources' to add directories to index, then run: kb index",
                "muted",
            )
        )
    else:
        if GLOBAL_CONFIG_FILE.exists():
            print_error(f"Global config already exists at {GLOBAL_CONFIG_FILE}")
            sys.exit(1)
        GLOBAL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        GLOBAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
        GLOBAL_CONFIG_FILE.write_text(GLOBAL_CONFIG_TEMPLATE)
        print(style(f"Created {GLOBAL_CONFIG_FILE}", "success"))
        print(label("Database", style(GLOBAL_DATA_DIR / "kb.db", "path")))
        print(style("Add sources with: kb add ~/notes ~/docs", "muted"))


def cmd_add(cfg: Config, dirs: list[str]):
    if not dirs:
        print_error("Usage: kb add <dir> [dir...]")
        sys.exit(1)

    for d in dirs:
        p = Path(d).expanduser().resolve()
        if not p.is_dir():
            print_error(f"Not a directory: {d}")
            sys.exit(1)

        if cfg.scope == "global":
            entry = str(p)
        else:
            try:
                entry = str(p.relative_to(cfg.config_dir))
            except ValueError:
                entry = str(p)

        if entry in cfg.sources:
            print(f"  {style('Already added:', 'warning')} {style(entry, 'path')}")
            continue

        cfg.sources.append(entry)
        print(f"  {style('Added:', 'success')} {style(entry, 'path')}")

    save_config(cfg)
    print(style(f"Saved {cfg.config_path}", "success"))


def cmd_remove(cfg: Config, dirs: list[str]):
    if not dirs:
        print_error("Usage: kb remove <dir> [dir...]")
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
            print(f"  {style('Removed:', 'success')} {style(entry, 'path')}")
        else:
            print(f"  {style('Not found:', 'warning')} {style(entry, 'path')}")

    save_config(cfg)
    print(style(f"Saved {cfg.config_path}", "success"))


def cmd_sources(cfg: Config):
    if not cfg.sources:
        print(style("No sources configured. Run: kb add <dir>", "warning"))
        return
    for s in cfg.sources:
        p = Path(s).expanduser() if cfg.scope == "global" else cfg.config_dir / s
        exists = p.is_dir()
        marker = " " if exists else style(" (missing)", "warning")
        print(f"  {style(s, 'path')}{marker}")


def cmd_allow(cfg: Config, files: list[str]):
    if not files:
        print_error("Usage: kb allow <file> [file...]")
        sys.exit(1)
    if not cfg.config_path:
        print_error("No config found. Run 'kb init' first.")
        sys.exit(1)

    for f in files:
        p = Path(f).expanduser().resolve()
        if not p.is_file():
            print_error(f"Not a file: {f}")
            sys.exit(1)

        if cfg.scope == "global":
            entry = str(p)
        else:
            try:
                entry = str(p.relative_to(cfg.config_dir))
            except ValueError:
                entry = str(p)

        if entry in cfg.allowed_large_files:
            print(f"  {style('Already allowed:', 'warning')} {style(entry, 'path')}")
            continue

        cfg.allowed_large_files.append(entry)
        print(f"  {style('Allowed:', 'success')} {style(entry, 'path')}")

    save_config(cfg)
    print(style(f"Saved {cfg.config_path}", "success"))


def cmd_index(cfg: Config, args: list[str]):
    no_size_limit = "--no-size-limit" in args
    dir_args = [a for a in args if a != "--no-size-limit"]

    if dir_args:
        dirs = [Path(a).resolve() for a in dir_args]
    elif cfg.source_paths:
        dirs = cfg.source_paths
    else:
        print(style("No sources configured. Either:", "warning"))
        print("  1. Run 'kb add <dir>' to add source directories")
        print("  2. Pass directories explicitly: kb index ~/docs ~/notes")
        sys.exit(1)

    for dir_path in dirs:
        if not dir_path.is_dir():
            print_error(f"Not a directory: {dir_path}")
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


# ---------------------------------------------------------------------------
# Output formatters for --csv and --md
# ---------------------------------------------------------------------------

_OUTPUT_FLAGS = ("--json", "--csv", "--md")


def _parse_output_format(args: list[str]) -> str | None:
    """Extract and remove output format flag from args list. Returns format or None."""
    for flag in _OUTPUT_FLAGS:
        if flag in args:
            args.remove(flag)
            return flag.lstrip("-")
    return None


def _format_csv(rows: list[dict], columns: list[str]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def _format_md_table(rows: list[dict], columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |"]
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in rows:
        cells = [
            str(row.get(c, "")).replace("|", "\\|").replace("\n", " ") for c in columns
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def cmd_search(
    query: str,
    cfg: Config,
    top_k: int = 5,
    threshold: float | None = None,
    output_format: str | None = None,
):
    try:
        result = search_core(query, cfg, top_k, threshold)
    except NoIndexError as e:
        print_error(str(e))
        sys.exit(1)

    if output_format == "json":
        print(json.dumps(result, ensure_ascii=False))
        return

    if output_format in ("csv", "md"):
        cols = [
            "rank",
            "doc_path",
            "heading",
            "similarity",
            "rrf_score",
            "sources",
            "text",
        ]
        rows = []
        for r in result["results"]:
            row = dict(r)
            row["sources"] = "+".join(row.get("sources", []))
            rows.append(row)
        formatter = _format_csv if output_format == "csv" else _format_md_table
        print(formatter(rows, cols), end="" if output_format == "csv" else "\n")
        return

    clean_query = result["query"]
    timing = result["timing_ms"]
    candidates = result["candidates"]

    if result.get("filters"):
        print(
            label(
                "Filters",
                ", ".join(f"{k}={v}" for k, v in result["filters"].items()),
            )
        )

    print(label("Query", f'"{clean_query}"'))
    hyde_tag = f"HyDE: {timing['hyde']}ms | " if timing.get("hyde") else ""
    expand_tag = f"Expand: {timing['expand']}ms | " if timing.get("expand") else ""
    print(
        style(
            f"{hyde_tag}{expand_tag}Embed: {timing['embed']}ms | "
            f"Vec: {timing['vec']}ms | FTS: {timing['fts']}ms",
            "metric",
        )
    )
    print(
        style(
            f"Candidates: {candidates['vec']} vec, {candidates['fts']} fts -> "
            f"{candidates['fused']} fused",
            "muted",
        )
    )

    if result.get("expansions"):
        lex = [e["text"] for e in result["expansions"] if e["type"] == "lex"]
        vec = [e["text"] for e in result["expansions"] if e["type"] == "vec"]
        parts = []
        if lex:
            parts.append(f"lex{lex}")
        if vec:
            parts.append(f"vec{vec}")
        print(label("Expansions", " ".join(parts)))

    print()

    for r in result["results"]:
        sim = (
            f"sim:{r['similarity']:.3f}" if r["similarity"] is not None else "fts-only"
        )
        source_tag = "+".join(r["sources"])
        print(
            result_header(
                r["rank"],
                r["doc_path"],
                f"{sim}, {source_tag}, rrf:{r['rrf_score']:.4f}",
            )
        )
        if r["heading"]:
            print(f"    {label('Section', style(r['heading'], 'heading'))}")
        preview = _best_snippet(r["text"] or "", clean_query).replace("\n", "\n    ")
        print(f"    {preview}")
        if r["text"] and len(r["text"]) > 500:
            total_chars = len(r["text"])
            print(f"    {style(f'({total_chars} chars total)', 'muted')}")
        print()


def cmd_fts(
    query: str,
    cfg: Config,
    top_k: int = 5,
    output_format: str | None = None,
):
    """FTS-only keyword search — no embedding, no API cost."""
    try:
        result = fts_core(query, cfg, top_k)
    except NoIndexError as e:
        print_error(str(e))
        sys.exit(1)
    except NoSearchTermsError as e:
        print_error(str(e))
        sys.exit(1)

    if output_format == "json":
        print(json.dumps(result, ensure_ascii=False))
        return

    if output_format in ("csv", "md"):
        cols = ["rank", "doc_path", "heading", "bm25", "text"]
        formatter = _format_csv if output_format == "csv" else _format_md_table
        print(
            formatter(result["results"], cols),
            end="" if output_format == "csv" else "\n",
        )
        return

    clean_query = result["query"]
    fts_ms = result["timing_ms"]["fts"]

    if result.get("filters"):
        print(
            label(
                "Filters",
                ", ".join(f"{k}={v}" for k, v in result["filters"].items()),
            )
        )

    print(label("Query", f'"{clean_query}"'))
    print(style(f"FTS: {fts_ms}ms | {len(result['results'])} results\n", "metric"))

    for r in result["results"]:
        bm25_str = f"bm25:{r['bm25']:.3f}"
        print(result_header(r["rank"], r["doc_path"], bm25_str))
        if r["heading"]:
            print(f"    {label('Section', style(r['heading'], 'heading'))}")
        preview = _best_snippet(r["text"] or "", clean_query).replace("\n", "\n    ")
        print(f"    {preview}")
        if r["text"] and len(r["text"]) > 500:
            total_chars = len(r["text"])
            print(f"    {style(f'({total_chars} chars total)', 'muted')}")
        print()


def cmd_ask(
    question: str,
    cfg: Config,
    top_k: int = 8,
    threshold: float | None = None,
    output_format: str | None = None,
):
    """Full RAG: hybrid retrieve -> filter -> LLM rerank -> confidence filter -> answer."""
    try:
        result = ask_core(question, cfg, top_k, threshold)
    except NoIndexError as e:
        print_error(str(e))
        sys.exit(1)

    if output_format == "json":
        out = {
            "question": result["question"],
            "answer": result["answer"],
            "model": result["model"],
            "bm25_shortcut": result["bm25_shortcut"],
            "rerank": result.get("rerank"),
            "filters": result.get("filters", {}),
            "timing_ms": result["timing_ms"],
            "tokens": result["tokens"],
            "cost": result.get("cost"),
            "sources": result["sources"],
            "result_count": result.get("result_count"),
            "filtered_count": result.get("filtered_count"),
        }
        if "expansions" in result:
            out["expanded"] = result.get("expanded", False)
            out["expansions"] = result["expansions"]
        print(json.dumps(out, ensure_ascii=False))
        return

    if output_format in ("csv", "md"):
        cols = ["rank", "doc_path", "heading"]
        header = (
            f"# Answer\n\n{result['answer']}\n\n# Sources\n\n"
            if output_format == "md"
            else ""
        )
        formatter = _format_csv if output_format == "csv" else _format_md_table
        body = formatter(result["sources"], cols)
        if output_format == "md":
            print(header + body)
        else:
            print(body, end="")
        return

    clean_question = result["question"]
    timing = result["timing_ms"]
    bm25_shortcut = result["bm25_shortcut"]
    rerank_info = result.get("rerank")

    if result.get("filters"):
        print(
            label(
                "Filters",
                ", ".join(f"{k}={v}" for k, v in result["filters"].items()),
            )
        )

    shortcut_tag = " (bm25 shortcut)" if bm25_shortcut else ""
    hyde_tag = f"hyde: {timing['hyde']}ms | " if timing.get("hyde") else ""
    expand_tag = f"expand: {timing['expand']}ms | " if timing.get("expand") else ""
    print(label("Q", clean_question))
    print(
        style(
            f"({hyde_tag}{expand_tag}embed: {timing['embed']}ms | "
            f"search: {timing['search']}ms | generate: "
            f"{timing['generate']}ms{shortcut_tag})",
            "metric",
        )
    )

    if result.get("expansions"):
        lex = [e["text"] for e in result["expansions"] if e["type"] == "lex"]
        vec = [e["text"] for e in result["expansions"] if e["type"] == "vec"]
        parts = []
        if lex:
            parts.append(f"lex{lex}")
        if vec:
            parts.append(f"vec{vec}")
        print(style(f"(expansions: {' '.join(parts)})", "muted"))

    if result["answer"] is None:
        if result.get("cost"):
            cost = result["cost"]
            known_tag = "" if cost.get("known", True) else " + unknown-price calls"
            print(
                style(
                    f"(estimated cost: {format_usd(cost['estimated_total_usd'])}"
                    f"{known_tag})",
                    "muted",
                )
            )
        print(style("\nNo relevant documents found.", "warning"))
        return

    if rerank_info:
        print(
            style(
                f"(rerank: {rerank_info['rerank_ms']:.0f}ms, "
                f"{rerank_info['prompt_tokens']}+"
                f"{rerank_info['completion_tokens']} tokens, "
                f"{rerank_info['input_count']} -> {rerank_info['output_count']})",
                "muted",
            )
        )

    print(style(f"(model: {result['model']})", "muted"))
    print(
        style(
            f"(tokens: {result['tokens']['prompt']} in / "
            f"{result['tokens']['completion']} out)",
            "muted",
        )
    )
    if result.get("cost"):
        cost = result["cost"]
        known_tag = "" if cost.get("known", True) else " + unknown-price calls"
        print(
            style(
                f"(estimated cost: {format_usd(cost['estimated_total_usd'])}"
                f"{known_tag})",
                "muted",
            )
        )
    print(
        style(
            f"(results: {result.get('result_count', '?')} retrieved, "
            f"{result.get('filtered_count', '?')} above threshold)\n",
            "muted",
        )
    )
    print(style(result["answer"], "answer"))
    print(style("\n--- Sources ---", "heading"))
    for src in result["sources"]:
        rank = style(f"[{src['rank']}]", "rank")
        path = style(src["doc_path"], "path")
        if src["heading"]:
            heading = style(src["heading"], "heading")
            print(f"  {rank} {path} > {heading}")
        else:
            print(f"  {rank} {path}")


def cmd_similar(file_arg: str, cfg: Config, top_k: int = 10):
    try:
        result = similar_core(file_arg, cfg, top_k)
    except NoIndexError as e:
        print_error(str(e))
        sys.exit(1)
    except FileNotIndexedError as e:
        print_error(str(e))
        if "not in index" in str(e).lower():
            print(style("Run 'kb index' to index it first.", "muted"))
        sys.exit(1)

    if not result["results"]:
        print(
            style(
                f"No similar documents found for {result['source']}.",
                "warning",
            )
        )
        return

    print(label("Documents similar to", style(result["source"], "path")) + "\n")
    for r in result["results"]:
        print(result_header(r["rank"], r["doc_path"], f"sim:{r['similarity']:.3f}"))
        if r["title"]:
            print(f"    {style(r['title'], 'heading')}")


def cmd_tag(cfg: Config, file_arg: str, new_tags: list[str]):
    if not cfg.db_path.exists():
        print_error("No index found. Run 'kb index' first.")
        sys.exit(1)

    conn = connect(cfg)
    doc_path = _resolve_doc_path(cfg, conn, file_arg)
    if not doc_path:
        print_error(f"File not in index: {file_arg}")
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
    print(label(f"Tags for {style(doc_path, 'path')}", ", ".join(sorted(existing))))
    conn.close()


def cmd_untag(cfg: Config, file_arg: str, remove_tags: list[str]):
    if not cfg.db_path.exists():
        print_error("No index found. Run 'kb index' first.")
        sys.exit(1)

    conn = connect(cfg)
    doc_path = _resolve_doc_path(cfg, conn, file_arg)
    if not doc_path:
        print_error(f"File not in index: {file_arg}")
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
        print(label(f"Tags for {style(doc_path, 'path')}", ", ".join(sorted(existing))))
    else:
        print(style(f"All tags removed from {doc_path}", "success"))
    conn.close()


def cmd_tags(cfg: Config):
    if not cfg.db_path.exists():
        print_error("No index found. Run 'kb index' first.")
        sys.exit(1)

    conn = connect(cfg)
    rows = conn.execute("SELECT tags FROM documents WHERE tags != ''").fetchall()
    conn.close()

    if not rows:
        print(style("No tagged documents.", "warning"))
        return

    counts: dict[str, int] = {}
    for r in rows:
        for tag in r["tags"].split(","):
            tag = tag.strip().lower()
            if tag:
                counts[tag] = counts.get(tag, 0) + 1

    print(style(f"{len(counts)} tags across {len(rows)} documents\n", "heading"))
    for tag, count in sorted(counts.items()):
        print(
            f"  {style(f'{tag:<30}', 'label')} "
            f"{style(count, 'metric')} doc{'s' if count != 1 else ''}"
        )


def cmd_stats(cfg: Config):
    result = stats_core(cfg)
    if "error" in result:
        print_error(result["error"])
        return

    db_size_kb = result["db_size_bytes"] / 1024
    print(label("DB", f"{style(result['db_path'], 'path')} ({db_size_kb:.1f} KB)"))
    print(label("Documents", style(result["doc_count"], "metric")), end="")
    if result["type_counts"]:
        parts = [f"{cnt} {t}" for t, cnt in result["type_counts"].items()]
        print(style(f" ({', '.join(parts)})", "muted"), end="")
    print()
    print(
        style(
            f"Chunks: {result['chunk_count']} | Vectors: {result['vec_count']} "
            f"| FTS entries: {result['fts_count']}",
            "metric",
        )
    )
    print(
        style(
            f"Total text: {result['total_chars']:,} chars "
            f"(~{result['total_chars'] // 4:,} tokens)",
            "metric",
        )
    )

    print(style("\nCapabilities:", "heading"))
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
    print(f"  {label('Supported formats', ', '.join(exts))}")

    missing = unavailable_formats()
    if missing:
        for ext, pkg in missing:
            print(f"  {style(ext + ':', 'warning')} unavailable (pip install {pkg})")

    print(style("\nDocuments:", "heading"))
    for doc in result["documents"]:
        h = doc["content_hash"][:8] if doc["content_hash"] else "n/a"
        type_tag = f" [{doc['type']}]" if doc["type"] != "markdown" else ""
        print(
            f"  {style(doc['path'], 'path')}: "
            f"{style(doc['chunk_count'], 'metric')} chunks "
            f"{style(f'[{h}]{type_tag}', 'muted')} ({doc['title']})"
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
        print_error(result["error"])
        return

    rows = result["documents"]
    if not rows:
        print(style("No documents indexed.", "warning"))
        return

    if full:
        print(style(f"{len(rows)} documents indexed\n", "heading"))
        for r in rows:
            path = r["path"]
            doc_type = r["type"] or "unknown"
            chunks = r["chunk_count"]
            size = r["size_bytes"]
            date = (r["indexed_at"] or "")[:10]
            print(
                f"  {style(f'{path:<50}', 'path')} {style(f'{doc_type:<12}', 'label')} "
                f"{style(f'{chunks:>3}', 'metric')} chunks  "
                f"{style(f'{_format_size(size):>10}', 'metric')}  "
                f"{style(date, 'muted')}"
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
        style(
            f"{len(rows)} documents indexed "
            f"({_format_size(total_size)}, {total_chunks} chunks)\n",
            "heading",
        )
    )
    for doc_type in sorted(
        type_stats, key=lambda t: type_stats[t]["count"], reverse=True
    ):
        s = type_stats[doc_type]
        count = s["count"]
        chunks = s["chunks"]
        size = _format_size(s["size"])
        print(
            f"  {style(f'{doc_type:<12}', 'label')} "
            f"{style(f'{count:>4}', 'metric')} docs  "
            f"{style(f'{chunks:>5}', 'metric')} chunks  "
            f"{style(f'{size:>10}', 'metric')}"
        )
    print(style("\nUse 'kb list --full' for per-file details.", "muted"))


def cmd_feedback(args: list[str]):
    """Submit or list feedback entries."""
    if "--list" in args:
        result = list_feedback_core()
        if not result["entries"]:
            print(style("No feedback entries.", "warning"))
            return
        print(style(f"{result['count']} feedback entries:\n", "heading"))
        for e in result["entries"]:
            sev = e.get("severity", "note")
            ts = e.get("timestamp", "?")
            msg = e.get("message", "")
            print(f"  {style(f'[{sev}]', 'rank')} {style(ts, 'muted')}")
            print(f"    {msg}")
            if e.get("tool"):
                print(f"    {label('tool', e['tool'])}")
            if e.get("agent_id"):
                print(f"    {label('agent', e['agent_id'])}")
            if e.get("error_trace"):
                print(f"    {label('trace', e['error_trace'])}")
            print()
        return

    # Parse flags
    message = ""
    tool = ""
    severity = "note"
    context = ""
    agent_id = ""
    error_trace = ""

    i = 0
    while i < len(args):
        if args[i] == "--tool" and i + 1 < len(args):
            tool = args[i + 1]
            i += 2
        elif args[i] == "--severity" and i + 1 < len(args):
            severity = args[i + 1]
            i += 2
        elif args[i] == "--context" and i + 1 < len(args):
            context = args[i + 1]
            i += 2
        elif args[i] == "--agent-id" and i + 1 < len(args):
            agent_id = args[i + 1]
            i += 2
        elif args[i] == "--error-trace" and i + 1 < len(args):
            error_trace = args[i + 1]
            i += 2
        elif args[i].startswith("--"):
            print_error(f"Unknown flag: {args[i]}")
            sys.exit(1)
        elif not message:
            message = args[i]
            i += 1
        else:
            print_error(f"Unexpected argument: {args[i]}")
            sys.exit(1)

    if not message:
        print_error(
            'Usage: kb feedback "message" [--tool T] [--severity bug|suggestion|note]'
        )
        sys.exit(1)

    try:
        entry = feedback_core(
            message,
            tool=tool,
            severity=severity,
            context=context,
            agent_id=agent_id,
            error_trace=error_trace,
        )
    except KBError as e:
        print_error(str(e))
        sys.exit(1)

    print(
        style(f"Feedback recorded [{entry['severity']}]: {entry['message']}", "success")
    )


def cmd_completion(shell: str):
    subcommands = (
        "init add remove sources index allow search fts ask similar "
        "tag untag tags stats reset list feedback version mcp completion"
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
      feedback)
        COMPREPLY=( $(compgen -W "--list --tool --severity --context --agent-id --error-trace" -- "$cur") )
        ;;
      search|ask)
        COMPREPLY=( $(compgen -W "--threshold --expand --no-expand --json --csv --md" -- "$cur") )
        ;;
      fts)
        COMPREPLY=( $(compgen -W "--json --csv --md" -- "$cur") )
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
            "complete -c kb -n '__fish_seen_subcommand_from feedback' "
            "-a '--list --tool --severity --context --agent-id --error-trace'"
        )
        print(
            "complete -c kb -n '__fish_seen_subcommand_from search ask' "
            "-a '--threshold --expand --no-expand --json --csv --md'"
        )
        print(
            "complete -c kb -n '__fish_seen_subcommand_from fts' -a '--json --csv --md'"
        )
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

    if cmd == "feedback":
        if len(args) > 1 and args[1] in ("-h", "--help"):
            print(
                'Usage: kb feedback "msg" [--tool T] [--severity bug|suggestion|note] '
                "[--context C] [--agent-id A] [--error-trace E]"
            )
            print("       kb feedback --list")
            sys.exit(0)
        cmd_feedback(args[1:])
        sys.exit(0)

    if cmd == "mcp":
        if len(args) > 1 and args[1] in ("-h", "--help"):
            print("Usage: kb mcp")
            print()
            print("Start the MCP (Model Context Protocol) server over stdio.")
            print("Used by Claude Desktop, Claude Code, and other MCP clients.")
            print()
            print("Claude Desktop config:")
            print('  {"mcpServers": {"kb": {"command": "kb-mcp"}}}')
            print()
            print("Claude Code:")
            print("  claude mcp add kb kb-mcp")
            sys.exit(0)
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
            print(
                'Usage: kb search "query" [k] [--threshold N] [--expand] [--json|--csv|--md]'
            )
            sys.exit(0 if sub_help else 1)
        threshold = None
        search_args = list(args[1:])
        out_fmt = _parse_output_format(search_args)
        if "--expand" in search_args:
            search_args.remove("--expand")
            cfg = copy(cfg)
            cfg.query_expand = True
        elif "--no-expand" in search_args:
            search_args.remove("--no-expand")
            cfg = copy(cfg)
            cfg.query_expand = False
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
            search_args[0], cfg, top_k, threshold=threshold, output_format=out_fmt
        )
    elif cmd == "fts":
        if len(args) < 2 or sub_help:
            print('Usage: kb fts "query" [k] [--json|--csv|--md]')
            sys.exit(0 if sub_help else 1)
        fts_args = list(args[1:])
        out_fmt = _parse_output_format(fts_args)
        top_k = int(fts_args[1]) if len(fts_args) > 1 else 5
        cmd_fts(fts_args[0], cfg, top_k, output_format=out_fmt)
    elif cmd == "ask":
        if len(args) < 2 or sub_help:
            print(
                'Usage: kb ask "question" [k] [--threshold N] [--expand] [--json|--csv|--md]'
            )
            sys.exit(0 if sub_help else 1)
        threshold = None
        ask_args = list(args[1:])
        out_fmt = _parse_output_format(ask_args)
        if "--expand" in ask_args:
            ask_args.remove("--expand")
            cfg = copy(cfg)
            cfg.query_expand = True
        elif "--no-expand" in ask_args:
            ask_args.remove("--no-expand")
            cfg = copy(cfg)
            cfg.query_expand = False
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
        cmd_ask(question, cfg, top_k, threshold=threshold, output_format=out_fmt)
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
