"""CLI entry point for kb."""

import sqlite3
import sys
import time
from pathlib import Path

from openai import OpenAI

from .chunk import CHONKIE_AVAILABLE
from .config import CONFIG_FILE, CONFIG_TEMPLATE, Config, find_config
from .db import connect, reset
from .embed import embed_batch, serialize_f32
from .filters import apply_filters, has_active_filters, parse_filters
from .ingest import PYMUPDF_AVAILABLE, index_directory
from .rerank import llm_rerank
from .search import fill_fts_only_results, fts_escape, rrf_fuse

USAGE = """\
kb — CLI knowledge base powered by sqlite-vec

Usage:
  kb init                        Create .kb.toml config in current directory
  kb index [DIR...]              Index sources from config (or explicit dirs)
  kb search "query" [k]          Hybrid semantic + keyword search (default k=5)
  kb ask "question" [k]          RAG: search + LLM rerank + answer (default k=8)
  kb stats                       Show index statistics
  kb reset                       Drop database and start fresh

Search filters (inline with query):
  file:articles/*.md             Glob filter on file path
  dt>"2026-02-01"                After date
  dt<"2026-02-14"                Before date
  +"keyword"                     Must contain
  -"keyword"                     Must not contain

Examples:
  kb init
  kb index
  kb index ~/notes ~/docs
  kb search 'file:articles/*.md cost optimization'
  kb ask 'dt>"2026-02-01" what deployment patterns?'
  kb search '+"docker" -"kubernetes" containers'
"""


def cmd_init():
    cfg_path = Path.cwd() / CONFIG_FILE
    if cfg_path.exists():
        print(f"{CONFIG_FILE} already exists at {cfg_path}")
        sys.exit(1)
    cfg_path.write_text(CONFIG_TEMPLATE)
    print(f"Created {cfg_path}")
    print("Edit 'sources' to add directories to index, then run: kb index")


def cmd_index(cfg: Config, args: list[str]):
    if args:
        dirs = [Path(a).resolve() for a in args]
    elif cfg.source_paths:
        dirs = cfg.source_paths
    else:
        print("No sources configured. Either:")
        print(f"  1. Run 'kb init' and add directories to {CONFIG_FILE}")
        print("  2. Pass directories explicitly: kb index ~/docs ~/notes")
        sys.exit(1)

    for dir_path in dirs:
        if not dir_path.is_dir():
            print(f"Not a directory: {dir_path}")
            sys.exit(1)
        index_directory(dir_path, cfg)


def cmd_search(query: str, cfg: Config, top_k: int = 5):
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
    resp = client.embeddings.create(model=cfg.embed_model, input=[clean_query], dimensions=cfg.embed_dims)
    query_emb = resp.data[0].embedding
    embed_ms = (time.time() - t0) * 1000

    retrieve_k = (top_k * 5) if has_filters else (top_k * 3)

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

    fuse_k = retrieve_k if has_filters else top_k
    results = rrf_fuse(vec_results, fts_results, fuse_k, cfg)
    fill_fts_only_results(conn, results)

    if has_filters:
        results = apply_filters(results, filters, conn)
        results = results[:top_k]

    print(f'Query: "{clean_query}"')
    print(f"Embed: {embed_ms:.0f}ms | Vec: {vec_ms:.1f}ms | FTS: {fts_ms:.1f}ms")
    print(f"Candidates: {len(vec_results)} vec, {len(fts_results)} fts -> {len(results)} fused\n")

    for i, r in enumerate(results):
        sim = f"sim:{r['similarity']:.3f}" if r["similarity"] is not None else "fts-only"
        sources = []
        if r["in_vec"]:
            sources.append("vec")
        if r["in_fts"]:
            sources.append("fts")
        source_tag = "+".join(sources)

        print(f"--- [{i + 1}] {r['doc_path']} ({sim}, {source_tag}, rrf:{r['rrf_score']:.4f}) ---")
        if r["heading"]:
            print(f"    Section: {r['heading']}")
        preview = (r["text"] or "")[:300].replace("\n", "\n    ")
        print(f"    {preview}")
        if r["text"] and len(r["text"]) > 300:
            print(f"    ... ({len(r['text'])} chars)")
        print()

    conn.close()


def cmd_ask(question: str, cfg: Config, top_k: int = 8):
    """Full RAG: hybrid retrieve -> filter -> LLM rerank -> confidence filter -> answer."""
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
    resp = client.embeddings.create(model=cfg.embed_model, input=[clean_question], dimensions=cfg.embed_dims)
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
        r for r in results if r["similarity"] is None or r["similarity"] >= cfg.min_similarity
    ]

    if not filtered:
        print(f"Q: {clean_question}")
        print(f"(embed: {embed_ms:.0f}ms | search: {vec_ms + fts_ms:.1f}ms)")
        print("\nNo relevant documents found.")
        conn.close()
        return

    context_parts = []
    sources = []
    for i, r in enumerate(filtered):
        sim = f"relevance: {r['similarity']:.2f}" if r["similarity"] is not None else "keyword match"
        label = f"[{i + 1}] {r['doc_path']}"
        if r["heading"]:
            label += f" > {r['heading']}"
        context_parts.append(f"--- Source {label} ({sim}) ---\n{r['text']}")
        if r["doc_path"] and r["doc_path"] not in sources:
            sources.append(r["doc_path"])

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
    print(f"(embed: {embed_ms:.0f}ms | search: {vec_ms + fts_ms:.1f}ms | generate: {gen_ms:.0f}ms)")
    print(f"(tokens: {tokens.prompt_tokens} in / {tokens.completion_tokens} out)")
    print(f"(results: {len(results)} retrieved, {len(filtered)} above threshold)\n")
    print(answer)
    print(f"\n--- Sources ---")
    for s in sources:
        print(f"  {s}")

    conn.close()


def cmd_stats(cfg: Config):
    if not cfg.db_path.exists():
        print("No index found.")
        return

    conn = connect(cfg)

    doc_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    chunk_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    vec_count = conn.execute("SELECT COUNT(*) FROM vec_chunks").fetchone()[0]
    total_chars = conn.execute("SELECT COALESCE(SUM(char_count), 0) FROM chunks").fetchone()[0]

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

    print(f"\nCapabilities:")
    print(f"  chonkie chunking: {'yes' if CHONKIE_AVAILABLE else 'no (pip install chonkie)'}")
    print(f"  PDF ingestion:    {'yes' if PYMUPDF_AVAILABLE else 'no (pip install pymupdf)'}")
    print(f"  LLM rerank:       yes (ask mode, top-{cfg.rerank_fetch_k} -> top-{cfg.rerank_top_k})")
    print(f'  Pre-search filters: yes (file:, dt>, dt<, +"kw", -"kw")')

    print(f"\nDocuments:")
    for row in conn.execute(
        "SELECT path, title, type, chunk_count, content_hash, indexed_at FROM documents ORDER BY path"
    ):
        h = row[4][:8] if row[4] else "n/a"
        type_tag = f" [{row[2]}]" if row[2] != "markdown" else ""
        print(f"  {row[0]}: {row[3]} chunks [{h}]{type_tag} ({row[1]})")

    conn.close()


def main():
    args = sys.argv[1:]

    if not args:
        print(USAGE)
        sys.exit(1)

    cmd = args[0]

    if cmd in ("-h", "--help", "help"):
        print(USAGE)
        sys.exit(0)

    if cmd == "init":
        cmd_init()
        sys.exit(0)

    # All other commands need config
    cfg = find_config()
    if cfg.config_dir:
        print(f"Config: {cfg.config_dir / CONFIG_FILE}")

    if cmd == "index":
        cmd_index(cfg, args[1:])
    elif cmd == "search":
        if len(args) < 2:
            print('Usage: kb search "query"')
            sys.exit(1)
        top_k = int(args[2]) if len(args) > 2 else 5
        cmd_search(args[1], cfg, top_k)
    elif cmd == "ask":
        if len(args) < 2:
            print('Usage: kb ask "question"')
            sys.exit(1)
        top_k = int(args[2]) if len(args) > 2 else 8
        cmd_ask(args[1], cfg, top_k)
    elif cmd == "stats":
        cmd_stats(cfg)
    elif cmd == "reset":
        reset(cfg.db_path)
    else:
        print(f"Unknown command: {cmd}")
        print(USAGE)
        sys.exit(1)
