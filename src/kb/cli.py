"""CLI entry point for kb."""

import sqlite3
import sys
import time
from pathlib import Path

from openai import OpenAI

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
from .embed import deserialize_f32, serialize_f32
from .filters import apply_filters, has_active_filters, parse_filters
from .extract import supported_extensions, unavailable_formats
from .ingest import index_directory
from .rerank import llm_rerank
from .search import fill_fts_only_results, fts_escape, rrf_fuse

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
  kb search "query" [k]          Hybrid semantic + keyword search (default k=5)
  kb ask "question" [k] [--threshold N]  RAG: search + rerank + answer (default k=8, threshold=0.001)
  kb similar <file> [k]          Find similar documents (no API call, default k=10)
  kb tag <file> tag1 [tag2...]   Add tags to a document
  kb untag <file> tag1 [tag2...]  Remove tags from a document
  kb tags                        List all tags with document counts
  kb list [--full]                List indexed documents (summary; --full for details)
  kb stats                       Show index statistics and supported formats
  kb reset                       Drop database and start fresh
  kb version                      Show version (also: kb v, kb --version)
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


def cmd_search(query: str, cfg: Config, top_k: int = 5, threshold: float | None = None):
    if threshold is not None:
        cfg.threshold = threshold
    if not cfg.db_path.exists():
        print("No index found. Run 'kb index' first.")
        sys.exit(1)

    conn = connect(cfg)
    client = OpenAI()

    clean_query, filters = parse_filters(query)
    has_filters = has_active_filters(filters)

    if has_filters:
        print(f"Filters: {', '.join(f'{k}={v}' for k, v in filters.items() if v)}")

    t0 = time.time()
    resp = client.embeddings.create(
        model=cfg.embed_model, input=[clean_query], dimensions=cfg.embed_dims
    )
    query_emb = resp.data[0].embedding
    embed_ms = (time.time() - t0) * 1000

    has_threshold = cfg.threshold > 0
    retrieve_k = (top_k * 5) if (has_filters or has_threshold) else (top_k * 3)

    t0 = time.time()
    vec_rows = conn.execute(
        "SELECT chunk_id, distance, chunk_text, doc_path, heading "
        "FROM vec_chunks WHERE embedding MATCH ? AND k = ? ORDER BY distance",
        (serialize_f32(query_emb), retrieve_k),
    ).fetchall()
    vec_results = [
        (r["chunk_id"], r["distance"], r["chunk_text"], r["doc_path"], r["heading"])
        for r in vec_rows
    ]
    vec_ms = (time.time() - t0) * 1000

    t0 = time.time()
    fts_q = fts_escape(clean_query)
    fts_results = []
    if fts_q:
        try:
            fts_rows = conn.execute(
                "SELECT rowid, rank FROM fts_chunks WHERE fts_chunks MATCH ? ORDER BY rank LIMIT ?",
                (fts_q, retrieve_k),
            ).fetchall()
            fts_results = [(r["rowid"], r["rank"]) for r in fts_rows]
        except sqlite3.OperationalError:
            pass
    fts_ms = (time.time() - t0) * 1000

    fuse_k = retrieve_k if (has_filters or has_threshold) else top_k
    results = rrf_fuse(vec_results, fts_results, fuse_k, cfg)
    fill_fts_only_results(conn, results)

    if has_filters:
        results = apply_filters(results, filters, conn)

    if has_threshold:
        results = [
            r
            for r in results
            if r["similarity"] is None or r["similarity"] >= cfg.threshold
        ]

    results = results[:top_k]

    print(f'Query: "{clean_query}"')
    print(f"Embed: {embed_ms:.0f}ms | Vec: {vec_ms:.1f}ms | FTS: {fts_ms:.1f}ms")
    print(
        f"Candidates: {len(vec_results)} vec, {len(fts_results)} fts -> {len(results)} fused\n"
    )

    for i, r in enumerate(results):
        sim = (
            f"sim:{r['similarity']:.3f}" if r["similarity"] is not None else "fts-only"
        )
        sources = []
        if r["in_vec"]:
            sources.append("vec")
        if r["in_fts"]:
            sources.append("fts")
        source_tag = "+".join(sources)

        print(
            f"--- [{i + 1}] {r['doc_path']} ({sim}, {source_tag}, rrf:{r['rrf_score']:.4f}) ---"
        )
        if r["heading"]:
            print(f"    Section: {r['heading']}")
        preview = (r["text"] or "")[:300].replace("\n", "\n    ")
        print(f"    {preview}")
        if r["text"] and len(r["text"]) > 300:
            print(f"    ... ({len(r['text'])} chars)")
        print()

    conn.close()


def cmd_ask(question: str, cfg: Config, top_k: int = 8, threshold: float | None = None):
    """Full RAG: hybrid retrieve -> filter -> LLM rerank -> confidence filter -> answer."""
    if threshold is not None:
        cfg.threshold = threshold
    if not cfg.db_path.exists():
        print("No index found. Run 'kb index' first.")
        sys.exit(1)

    conn = connect(cfg)
    client = OpenAI()

    clean_question, filters = parse_filters(question)
    has_filters = has_active_filters(filters)

    if has_filters:
        print(f"Filters: {', '.join(f'{k}={v}' for k, v in filters.items() if v)}")

    t0 = time.time()
    resp = client.embeddings.create(
        model=cfg.embed_model, input=[clean_question], dimensions=cfg.embed_dims
    )
    query_emb = resp.data[0].embedding
    embed_ms = (time.time() - t0) * 1000

    retrieve_k = max(cfg.rerank_fetch_k, top_k * 3)

    t0 = time.time()
    vec_rows = conn.execute(
        "SELECT chunk_id, distance, chunk_text, doc_path, heading "
        "FROM vec_chunks WHERE embedding MATCH ? AND k = ? ORDER BY distance",
        (serialize_f32(query_emb), retrieve_k),
    ).fetchall()
    vec_results = [
        (r["chunk_id"], r["distance"], r["chunk_text"], r["doc_path"], r["heading"])
        for r in vec_rows
    ]
    vec_ms = (time.time() - t0) * 1000

    t0 = time.time()
    fts_q = fts_escape(clean_question)
    fts_results = []
    if fts_q:
        try:
            fts_rows = conn.execute(
                "SELECT rowid, rank FROM fts_chunks WHERE fts_chunks MATCH ? ORDER BY rank LIMIT ?",
                (fts_q, retrieve_k),
            ).fetchall()
            fts_results = [(r["rowid"], r["rank"]) for r in fts_rows]
        except sqlite3.OperationalError:
            pass
    fts_ms = (time.time() - t0) * 1000

    results = rrf_fuse(vec_results, fts_results, cfg.rerank_fetch_k, cfg)
    fill_fts_only_results(conn, results)

    if has_filters:
        results = apply_filters(results, filters, conn)

    if len(results) > cfg.rerank_top_k:
        results = llm_rerank(client, clean_question, results, cfg)

    filtered = [
        r
        for r in results
        if r["similarity"] is None or r["similarity"] >= cfg.threshold
    ]

    if not filtered:
        print(f"Q: {clean_question}")
        print(f"(embed: {embed_ms:.0f}ms | search: {vec_ms + fts_ms:.1f}ms)")
        print("\nNo relevant documents found.")
        conn.close()
        return

    context_parts = []
    source_entries = []  # (doc_path, display_heading) for dedup
    for i, r in enumerate(filtered):
        sim = (
            f"relevance: {r['similarity']:.2f}"
            if r["similarity"] is not None
            else "keyword match"
        )
        # Prefer heading_ancestry (full breadcrumb) from chunks table
        ancestry = None
        row = conn.execute(
            "SELECT heading_ancestry FROM chunks WHERE id = ?",
            (r["chunk_id"],),
        ).fetchone()
        if row and row["heading_ancestry"]:
            ancestry = row["heading_ancestry"]

        display_heading = ancestry or r["heading"] or ""
        label = f"[{i + 1}] {r['doc_path']}"
        if display_heading:
            label += f" > {display_heading}"
        context_parts.append(f"--- Source {label} ({sim}) ---\n{r['text']}")
        source_key = (r["doc_path"], display_heading)
        if r["doc_path"] and source_key not in source_entries:
            source_entries.append(source_key)

    context = "\n\n".join(context_parts)

    t0 = time.time()
    chat_resp = client.chat.completions.create(
        model=cfg.chat_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You answer questions based on the provided context from a personal knowledge base. "
                    "Be direct and concise. If the context doesn't contain enough information, say so. "
                    "Cite sources by their number [1], [2], etc. when referencing specific information."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\n---\nQuestion: {clean_question}",
            },
        ],
        temperature=0.3,
        max_tokens=1000,
    )
    answer = chat_resp.choices[0].message.content
    gen_ms = (time.time() - t0) * 1000
    tokens = chat_resp.usage

    print(f"Q: {clean_question}")
    print(
        f"(embed: {embed_ms:.0f}ms | search: {vec_ms + fts_ms:.1f}ms | generate: {gen_ms:.0f}ms)"
    )
    print(f"(model: {cfg.chat_model})")
    print(f"(tokens: {tokens.prompt_tokens} in / {tokens.completion_tokens} out)")
    print(f"(results: {len(results)} retrieved, {len(filtered)} above threshold)\n")
    print(answer)
    print("\n--- Sources ---")
    for i, (path, heading) in enumerate(source_entries, 1):
        if heading:
            print(f"  [{i}] {path} > {heading}")
        else:
            print(f"  [{i}] {path}")

    conn.close()


def _resolve_doc_path(
    cfg: Config, conn: sqlite3.Connection, file_arg: str
) -> str | None:
    """Resolve a file argument to a doc_path in the DB."""
    p = Path(file_arg).expanduser().resolve()

    # Try relative to config_dir
    if cfg.config_dir:
        try:
            rel = str(p.relative_to(cfg.config_dir))
            row = conn.execute(
                "SELECT path FROM documents WHERE path = ?", (rel,)
            ).fetchone()
            if row:
                return row["path"]
        except ValueError:
            pass

    # Try as-is
    row = conn.execute(
        "SELECT path FROM documents WHERE path = ?", (file_arg,)
    ).fetchone()
    if row:
        return row["path"]

    # Try matching suffix
    rows = conn.execute("SELECT path FROM documents").fetchall()
    for r in rows:
        if r["path"].endswith(file_arg) or file_arg.endswith(r["path"]):
            return r["path"]

    return None


def cmd_similar(file_arg: str, cfg: Config, top_k: int = 10):
    if not cfg.db_path.exists():
        print("No index found. Run 'kb index' first.")
        sys.exit(1)

    conn = connect(cfg)

    doc_path = _resolve_doc_path(cfg, conn, file_arg)
    if not doc_path:
        print(f"File not in index: {file_arg}")
        print("Run 'kb index' to index it first.")
        conn.close()
        sys.exit(1)

    # Get chunk IDs for this document
    chunk_ids = [
        r["id"]
        for r in conn.execute(
            "SELECT c.id FROM chunks c JOIN documents d ON d.id = c.doc_id WHERE d.path = ?",
            (doc_path,),
        ).fetchall()
    ]

    if not chunk_ids:
        print(f"No chunks found for {doc_path}.")
        conn.close()
        sys.exit(1)

    # Read embeddings and average them
    embeddings = []
    for cid in chunk_ids:
        row = conn.execute(
            "SELECT embedding FROM vec_chunks WHERE chunk_id = ?", (cid,)
        ).fetchone()
        if row:
            embeddings.append(deserialize_f32(row["embedding"]))

    if not embeddings:
        print(f"No embeddings found for {doc_path}.")
        conn.close()
        sys.exit(1)

    # Average into a single vector
    dims = len(embeddings[0])
    avg = [sum(e[d] for e in embeddings) / len(embeddings) for d in range(dims)]

    # KNN query — fetch extra to account for self-document filtering
    fetch_k = top_k + len(chunk_ids) + 5
    rows = conn.execute(
        "SELECT chunk_id, distance, doc_path FROM vec_chunks "
        "WHERE embedding MATCH ? AND k = ? ORDER BY distance",
        (serialize_f32(avg), fetch_k),
    ).fetchall()

    # Aggregate by document, skip source document
    best_per_doc: dict[str, float] = {}
    for r in rows:
        dp = r["doc_path"]
        if dp == doc_path:
            continue
        dist = r["distance"]
        if dp not in best_per_doc or dist < best_per_doc[dp]:
            best_per_doc[dp] = dist

    # Sort by similarity (1 - distance) descending
    ranked = sorted(best_per_doc.items(), key=lambda x: x[1])[:top_k]

    if not ranked:
        print(f"No similar documents found for {doc_path}.")
        conn.close()
        return

    # Get titles
    titles = {}
    for r in conn.execute("SELECT path, title FROM documents").fetchall():
        titles[r["path"]] = r["title"]

    print(f"Documents similar to: {doc_path}\n")
    for i, (dp, dist) in enumerate(ranked, 1):
        sim = 1 - dist
        title = titles.get(dp, "")
        print(f"--- [{i}] {dp} (sim:{sim:.3f}) ---")
        if title:
            print(f"    {title}")

    conn.close()


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
    if not cfg.db_path.exists():
        print("No index found.")
        return

    conn = connect(cfg)

    doc_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    chunk_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    vec_count = conn.execute("SELECT COUNT(*) FROM vec_chunks").fetchone()[0]
    total_chars = conn.execute(
        "SELECT COALESCE(SUM(char_count), 0) FROM chunks"
    ).fetchone()[0]

    fts_count = 0
    try:
        fts_count = conn.execute("SELECT COUNT(*) FROM fts_chunks").fetchone()[0]
    except sqlite3.OperationalError:
        pass

    type_counts = conn.execute(
        "SELECT type, COUNT(*) as cnt FROM documents GROUP BY type ORDER BY type"
    ).fetchall()

    print(f"DB: {cfg.db_path} ({cfg.db_path.stat().st_size / 1024:.1f} KB)")
    print(f"Documents: {doc_count}", end="")
    if type_counts:
        parts = [f"{r[1]} {r[0]}" for r in type_counts]
        print(f" ({', '.join(parts)})", end="")
    print()
    print(f"Chunks: {chunk_count} | Vectors: {vec_count} | FTS entries: {fts_count}")
    print(f"Total text: {total_chars:,} chars (~{total_chars // 4:,} tokens)")

    print("\nCapabilities:")
    print(
        f"  chonkie chunking:   {'yes' if CHONKIE_AVAILABLE else 'no (pip install chonkie)'}"
    )
    print(
        f"  LLM rerank:         yes (ask mode, top-{cfg.rerank_fetch_k} -> top-{cfg.rerank_top_k})"
    )
    print('  Pre-search filters: yes (file:, type:, tag:, dt>, dt<, +"kw", -"kw")')
    print(
        f"  Index code files:   {'yes' if cfg.index_code else 'no (set index_code = true)'}"
    )

    exts = sorted(supported_extensions(include_code=cfg.index_code))
    print(f"  Supported formats:  {', '.join(exts)}")

    missing = unavailable_formats()
    if missing:
        for ext, pkg in missing:
            print(f"  {ext}: unavailable (pip install {pkg})")

    print("\nDocuments:")
    for row in conn.execute(
        "SELECT path, title, type, chunk_count, content_hash, indexed_at FROM documents ORDER BY path"
    ):
        h = row[4][:8] if row[4] else "n/a"
        type_tag = f" [{row[2]}]" if row[2] != "markdown" else ""
        print(f"  {row[0]}: {row[3]} chunks [{h}]{type_tag} ({row[1]})")

    conn.close()


def _format_size(size: int) -> str:
    if size >= 1_000_000:
        return f"{size / 1_000_000:.1f} MB"
    if size >= 1_000:
        return f"{size / 1_000:.1f} KB"
    return f"{size} B"


def cmd_list(cfg: Config, full: bool = False):
    if not cfg.db_path.exists():
        print("No index found. Run 'kb index' first.")
        return

    conn = connect(cfg)
    rows = conn.execute(
        "SELECT path, type, chunk_count, size_bytes, indexed_at "
        "FROM documents ORDER BY path"
    ).fetchall()
    conn.close()

    if not rows:
        print("No documents indexed.")
        return

    if full:
        print(f"{len(rows)} documents indexed\n")
        for r in rows:
            path = r["path"]
            doc_type = r["type"] or "unknown"
            chunks = r["chunk_count"] or 0
            size = r["size_bytes"] or 0
            date = (r["indexed_at"] or "")[:10]
            print(
                f"  {path:<50} {doc_type:<12} {chunks:>3} chunks  {_format_size(size):>10}  {date}"
            )
        return

    # Summary view: count and size by type
    type_stats: dict[str, dict] = {}
    total_size = 0
    total_chunks = 0
    for r in rows:
        doc_type = r["type"] or "unknown"
        size = r["size_bytes"] or 0
        chunks = r["chunk_count"] or 0
        total_size += size
        total_chunks += chunks
        if doc_type not in type_stats:
            type_stats[doc_type] = {"count": 0, "size": 0, "chunks": 0}
        type_stats[doc_type]["count"] += 1
        type_stats[doc_type]["size"] += size
        type_stats[doc_type]["chunks"] += chunks

    print(
        f"{len(rows)} documents indexed ({_format_size(total_size)}, {total_chunks} chunks)\n"
    )
    for doc_type in sorted(
        type_stats, key=lambda t: type_stats[t]["count"], reverse=True
    ):
        s = type_stats[doc_type]
        print(
            f"  {doc_type:<12} {s['count']:>4} docs  {s['chunks']:>5} chunks  {_format_size(s['size']):>10}"
        )
    print("\nUse 'kb list --full' for per-file details.")


def cmd_completion(shell: str):
    subcommands = "init add remove sources index allow search ask similar tag untag tags stats reset list version completion"

    if shell == "zsh":
        print(f"""\
_kb() {{
  local -a commands
  commands=({subcommands})
  _arguments '1:command:({" ".join(subcommands.split())})' '*:file:_files'
}}
compdef _kb kb""")
    elif shell == "bash":
        print(f"""\
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
      ask)
        COMPREPLY=( $(compgen -W "--threshold" -- "$cur") )
        ;;
      completion)
        COMPREPLY=( $(compgen -W "zsh bash fish" -- "$cur") )
        ;;
    esac
  fi
}}
complete -F _kb kb""")
    elif shell == "fish":
        cmds = subcommands.split()
        print("# Fish completions for kb")
        for c in cmds:
            print(f"complete -c kb -n '__fish_use_subcommand' -a {c}")
        print(
            "complete -c kb -n '__fish_seen_subcommand_from add remove index allow' -F"
        )
        print("complete -c kb -n '__fish_seen_subcommand_from init' -a '--project'")
        print("complete -c kb -n '__fish_seen_subcommand_from ask' -a '--threshold'")
        print(
            "complete -c kb -n '__fish_seen_subcommand_from completion' -a 'zsh bash fish'"
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
            print('Usage: kb search "query" [k] [--threshold N]')
            sys.exit(0 if sub_help else 1)
        threshold = None
        search_args = list(args[1:])
        if "--threshold" in search_args:
            ti = search_args.index("--threshold")
            if ti + 1 < len(search_args):
                threshold = float(search_args[ti + 1])
                del search_args[ti : ti + 2]
            else:
                print("--threshold requires a value")
                sys.exit(1)
        top_k = int(search_args[1]) if len(search_args) > 1 else 5
        cmd_search(search_args[0], cfg, top_k, threshold=threshold)
    elif cmd == "ask":
        if len(args) < 2 or sub_help:
            print('Usage: kb ask "question" [k] [--threshold N]')
            sys.exit(0 if sub_help else 1)
        threshold = None
        ask_args = list(args[1:])
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
        cmd_ask(question, cfg, top_k, threshold=threshold)
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
