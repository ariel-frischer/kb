"""Core API functions for kb. Used by both CLI and MCP server."""

import sqlite3
import time
from copy import copy
from pathlib import Path

from openai import OpenAI

from .config import Config
from .db import connect
from .embed import deserialize_f32, embed_batch, serialize_f32
from .expand import expand_query
from .filters import apply_filters, has_active_filters, parse_filters
from .hyde import generate_hyde_passage
from .rerank import rerank
from .search import (
    fill_fts_only_results,
    fts_escape,
    multi_rrf_fuse,
    normalize_fts_list,
    normalize_vec_list,
    rrf_fuse,
    run_fts_query,
    run_vec_query,
)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class KBError(Exception):
    """Base error for kb operations."""


class NoIndexError(KBError):
    """Raised when the database doesn't exist."""


class NoSearchTermsError(KBError):
    """Raised when query has no searchable terms after filter parsing."""


class FileNotIndexedError(KBError):
    """Raised when a file is not found in the index."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_doc_path(
    cfg: Config, conn: sqlite3.Connection, file_arg: str
) -> str | None:
    """Resolve a file argument to a doc_path in the DB."""
    p = Path(file_arg).expanduser().resolve()

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

    row = conn.execute(
        "SELECT path FROM documents WHERE path = ?", (file_arg,)
    ).fetchone()
    if row:
        return row["path"]

    rows = conn.execute("SELECT path FROM documents").fetchall()
    for r in rows:
        if r["path"].endswith(file_arg) or file_arg.endswith(r["path"]):
            return r["path"]

    return None


def _require_index(cfg: Config) -> None:
    if not cfg.db_path.exists():
        raise NoIndexError("No index found. Run 'kb index' first.")


# ---------------------------------------------------------------------------
# Core functions — return dicts, never print or sys.exit
# ---------------------------------------------------------------------------


def search_core(
    query: str,
    cfg: Config,
    top_k: int = 5,
    threshold: float | None = None,
) -> dict:
    if threshold is not None:
        cfg = copy(cfg)
        cfg.search_threshold = threshold

    _require_index(cfg)

    conn = connect(cfg)
    client = OpenAI()

    clean_query, filters = parse_filters(query)
    has_filters = has_active_filters(filters)

    hyde_passage = None
    hyde_ms = 0.0
    expand_ms = 0.0
    expansions: list[dict] = []
    embed_input = clean_query
    if cfg.hyde_enabled:
        hyde_passage, hyde_ms = generate_hyde_passage(clean_query, client, cfg)
        if hyde_passage:
            embed_input = hyde_passage

    has_threshold = cfg.search_threshold > 0
    retrieve_k = (top_k * 5) if has_filters else (top_k * 3)

    if cfg.query_expand:
        expansions, expand_ms = expand_query(client, clean_query, cfg)
        lex_exps = [e for e in expansions if e["type"] == "lex"]
        vec_exps = [e for e in expansions if e["type"] == "vec"]

        # Batch embed: primary + all vec expansions (single API call)
        t0 = time.time()
        all_vec_texts = [embed_input] + [e["text"] for e in vec_exps]
        all_embeddings = embed_batch(client, all_vec_texts, cfg)
        embed_ms = (time.time() - t0) * 1000

        # Run all queries
        t0 = time.time()
        primary_vec = run_vec_query(conn, serialize_f32(all_embeddings[0]), retrieve_k)
        exp_vec = [
            run_vec_query(conn, serialize_f32(emb), retrieve_k)
            for emb in all_embeddings[1:]
        ]
        vec_ms = (time.time() - t0) * 1000

        t0 = time.time()
        primary_fts = run_fts_query(conn, clean_query, retrieve_k)
        exp_fts = [run_fts_query(conn, e["text"], retrieve_k) for e in lex_exps]
        fts_ms = (time.time() - t0) * 1000

        # Normalize and fuse all lists
        all_lists = [
            normalize_vec_list(primary_vec),
            normalize_fts_list(primary_fts),
        ]
        all_lists += [normalize_vec_list(r) for r in exp_vec]
        all_lists += [normalize_fts_list(r) for r in exp_fts]
        weights = [2.0, 2.0] + [1.0] * (len(all_lists) - 2)

        fuse_k = retrieve_k if has_filters else top_k
        results = multi_rrf_fuse(all_lists, weights, fuse_k, cfg.rrf_k)
        fill_fts_only_results(conn, results)

        vec_count = len(primary_vec)
        fts_count = len(primary_fts)
    else:
        t0 = time.time()
        resp = client.embeddings.create(
            model=cfg.embed_model, input=[embed_input], dimensions=cfg.embed_dims
        )
        query_emb = resp.data[0].embedding
        embed_ms = (time.time() - t0) * 1000

        t0 = time.time()
        vec_rows = conn.execute(
            "SELECT chunk_id, distance, chunk_text, doc_path, heading "
            "FROM vec_chunks WHERE embedding MATCH ? AND k = ? ORDER BY distance",
            (serialize_f32(query_emb), retrieve_k),
        ).fetchall()
        vec_results = [
            (
                r["chunk_id"],
                r["distance"],
                r["chunk_text"],
                r["doc_path"],
                r["heading"],
            )
            for r in vec_rows
        ]
        vec_ms = (time.time() - t0) * 1000

        t0 = time.time()
        fts_q = fts_escape(clean_query)
        fts_results: list[tuple] = []
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

        vec_count = len(vec_results)
        fts_count = len(fts_results)

    if has_filters:
        results = apply_filters(results, filters, conn)

    results = results[:top_k]

    if has_threshold:
        results = [
            r
            for r in results
            if r["similarity"] is None or r["similarity"] >= cfg.search_threshold
        ]

    conn.close()

    timing = {
        "hyde": round(hyde_ms),
        "embed": round(embed_ms),
        "vec": round(vec_ms),
        "fts": round(fts_ms),
    }
    if cfg.query_expand:
        timing["expand"] = round(expand_ms)

    out: dict = {
        "query": clean_query,
        "filters": {k: v for k, v in filters.items() if v} if has_filters else {},
        "timing_ms": timing,
        "candidates": {
            "vec": vec_count,
            "fts": fts_count,
            "fused": len(results),
        },
        "results": [
            {
                "rank": i + 1,
                "doc_path": r["doc_path"],
                "heading": r["heading"],
                "similarity": r["similarity"],
                "fts_rank": r.get("fts_rank"),
                "rrf_score": round(r["rrf_score"], 6),
                "sources": [s for s in ["vec", "fts"] if r[f"in_{s}"]],
                "text": r["text"],
            }
            for i, r in enumerate(results)
        ],
    }
    if cfg.query_expand:
        out["expanded"] = bool(expansions)
        out["expansions"] = expansions
    return out


def fts_core(
    query: str,
    cfg: Config,
    top_k: int = 5,
) -> dict:
    _require_index(cfg)

    conn = connect(cfg)

    clean_query, filters = parse_filters(query)
    has_filters = has_active_filters(filters)

    fts_q = fts_escape(clean_query)
    if not fts_q:
        conn.close()
        raise NoSearchTermsError("No searchable terms in query.")

    t0 = time.time()
    try:
        fts_rows = conn.execute(
            "SELECT rowid, rank FROM fts_chunks WHERE fts_chunks MATCH ? ORDER BY rank LIMIT ?",
            (fts_q, top_k * 5 if has_filters else top_k),
        ).fetchall()
    except sqlite3.OperationalError:
        conn.close()
        raise NoIndexError("FTS index not available. Re-run 'kb index'.")
    fts_ms = (time.time() - t0) * 1000

    results = []
    for chunk_id, fts_rank in [(r["rowid"], r["rank"]) for r in fts_rows]:
        abs_rank = abs(fts_rank)
        norm_bm25 = abs_rank / (1.0 + abs_rank)
        results.append(
            {
                "chunk_id": chunk_id,
                "rrf_score": norm_bm25,
                "distance": None,
                "similarity": None,
                "fts_rank": fts_rank,
                "text": None,
                "doc_path": None,
                "heading": None,
                "in_fts": True,
                "in_vec": False,
            }
        )
    fill_fts_only_results(conn, results)

    if has_filters:
        results = apply_filters(results, filters, conn)

    results = results[:top_k]
    conn.close()

    return {
        "query": clean_query,
        "filters": {k: v for k, v in filters.items() if v} if has_filters else {},
        "timing_ms": {"fts": round(fts_ms)},
        "results": [
            {
                "rank": i + 1,
                "doc_path": r["doc_path"],
                "heading": r["heading"],
                "bm25": round(r["rrf_score"], 4),
                "fts_rank": r["fts_rank"],
                "text": r["text"],
            }
            for i, r in enumerate(results)
        ],
    }


def ask_core(
    question: str,
    cfg: Config,
    top_k: int = 8,
    threshold: float | None = None,
) -> dict:
    if threshold is not None:
        cfg = copy(cfg)
        cfg.ask_threshold = threshold

    _require_index(cfg)

    conn = connect(cfg)
    client = OpenAI()

    clean_question, filters = parse_filters(question)
    has_filters = has_active_filters(filters)

    # BM25 shortcut
    bm25_shortcut = False
    fts_q = fts_escape(clean_question)
    if fts_q and not has_filters:
        try:
            probe = conn.execute(
                "SELECT rowid, rank FROM fts_chunks WHERE fts_chunks MATCH ? ORDER BY rank LIMIT 2",
                (fts_q,),
            ).fetchall()
            if probe:
                top_norm = abs(probe[0]["rank"]) / (1.0 + abs(probe[0]["rank"]))
                second_norm = (
                    abs(probe[1]["rank"]) / (1.0 + abs(probe[1]["rank"]))
                    if len(probe) > 1
                    else 0.0
                )
                if top_norm >= 0.85 and (top_norm - second_norm) >= 0.15:
                    bm25_shortcut = True
        except sqlite3.OperationalError:
            pass

    rerank_info = None
    hyde_ms = 0.0
    expand_ms = 0.0
    expansions: list[dict] = []

    if bm25_shortcut:
        t0 = time.time()
        fts_rows = conn.execute(
            "SELECT rowid, rank FROM fts_chunks WHERE fts_chunks MATCH ? ORDER BY rank LIMIT ?",
            (fts_q, top_k),
        ).fetchall()
        fts_results = [(r["rowid"], r["rank"]) for r in fts_rows]
        search_ms = (time.time() - t0) * 1000
        embed_ms = 0.0

        results = []
        for chunk_id, fts_rank in fts_results:
            abs_rank = abs(fts_rank)
            norm_bm25 = abs_rank / (1.0 + abs_rank)
            results.append(
                {
                    "chunk_id": chunk_id,
                    "rrf_score": norm_bm25,
                    "distance": None,
                    "similarity": None,
                    "fts_rank": fts_rank,
                    "text": None,
                    "doc_path": None,
                    "heading": None,
                    "in_fts": True,
                    "in_vec": False,
                }
            )
        fill_fts_only_results(conn, results)
    else:
        embed_input = clean_question
        if cfg.hyde_enabled:
            hyde_passage, hyde_ms = generate_hyde_passage(clean_question, client, cfg)
            if hyde_passage:
                embed_input = hyde_passage

        retrieve_k = max(cfg.rerank_fetch_k, top_k * 3)

        if cfg.query_expand:
            expansions, expand_ms = expand_query(client, clean_question, cfg)
            lex_exps = [e for e in expansions if e["type"] == "lex"]
            vec_exps = [e for e in expansions if e["type"] == "vec"]

            t0 = time.time()
            all_vec_texts = [embed_input] + [e["text"] for e in vec_exps]
            all_embeddings = embed_batch(client, all_vec_texts, cfg)
            embed_ms = (time.time() - t0) * 1000

            t0 = time.time()
            primary_vec = run_vec_query(
                conn, serialize_f32(all_embeddings[0]), retrieve_k
            )
            exp_vec = [
                run_vec_query(conn, serialize_f32(emb), retrieve_k)
                for emb in all_embeddings[1:]
            ]
            primary_fts = run_fts_query(conn, clean_question, retrieve_k)
            exp_fts = [run_fts_query(conn, e["text"], retrieve_k) for e in lex_exps]
            search_ms = (time.time() - t0) * 1000

            all_lists = [
                normalize_vec_list(primary_vec),
                normalize_fts_list(primary_fts),
            ]
            all_lists += [normalize_vec_list(r) for r in exp_vec]
            all_lists += [normalize_fts_list(r) for r in exp_fts]
            weights = [2.0, 2.0] + [1.0] * (len(all_lists) - 2)

            results = multi_rrf_fuse(all_lists, weights, cfg.rerank_fetch_k, cfg.rrf_k)
            fill_fts_only_results(conn, results)
        else:
            t0 = time.time()
            resp = client.embeddings.create(
                model=cfg.embed_model,
                input=[embed_input],
                dimensions=cfg.embed_dims,
            )
            query_emb = resp.data[0].embedding
            embed_ms = (time.time() - t0) * 1000

            t0 = time.time()
            vec_rows = conn.execute(
                "SELECT chunk_id, distance, chunk_text, doc_path, heading "
                "FROM vec_chunks WHERE embedding MATCH ? AND k = ? ORDER BY distance",
                (serialize_f32(query_emb), retrieve_k),
            ).fetchall()
            vec_results = [
                (
                    r["chunk_id"],
                    r["distance"],
                    r["chunk_text"],
                    r["doc_path"],
                    r["heading"],
                )
                for r in vec_rows
            ]

            fts_results_ask: list[tuple] = []
            if fts_q:
                try:
                    fts_rows = conn.execute(
                        "SELECT rowid, rank FROM fts_chunks WHERE fts_chunks MATCH ? ORDER BY rank LIMIT ?",
                        (fts_q, retrieve_k),
                    ).fetchall()
                    fts_results_ask = [(r["rowid"], r["rank"]) for r in fts_rows]
                except sqlite3.OperationalError:
                    pass
            search_ms = (time.time() - t0) * 1000

            results = rrf_fuse(vec_results, fts_results_ask, cfg.rerank_fetch_k, cfg)
            fill_fts_only_results(conn, results)

        if has_filters:
            results = apply_filters(results, filters, conn)

        if len(results) > cfg.rerank_top_k:
            results, rerank_info = rerank(client, clean_question, results, cfg)

    filtered = [
        r
        for r in results
        if r["similarity"] is None or r["similarity"] >= cfg.ask_threshold
    ]

    active_filters = {k: v for k, v in filters.items() if v} if has_filters else {}

    timing = {
        "hyde": round(hyde_ms),
        "embed": round(embed_ms),
        "search": round(search_ms),
        "generate": 0,
    }
    if cfg.query_expand:
        timing["expand"] = round(expand_ms)

    if not filtered:
        conn.close()
        out: dict = {
            "question": clean_question,
            "filters": active_filters,
            "answer": None,
            "model": cfg.chat_model,
            "bm25_shortcut": bm25_shortcut,
            "rerank": rerank_info,
            "timing_ms": timing,
            "tokens": {"prompt": 0, "completion": 0},
            "sources": [],
        }
        if cfg.query_expand:
            out["expanded"] = bool(expansions)
            out["expansions"] = expansions
        return out

    context_parts = []
    source_entries = []
    for i, r in enumerate(filtered):
        sim = (
            f"relevance: {r['similarity']:.2f}"
            if r["similarity"] is not None
            else "keyword match"
        )
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

    conn.close()

    timing["generate"] = round(gen_ms)

    out = {
        "question": clean_question,
        "filters": active_filters,
        "answer": answer,
        "model": cfg.chat_model,
        "bm25_shortcut": bm25_shortcut,
        "rerank": rerank_info,
        "timing_ms": timing,
        "tokens": {
            "prompt": tokens.prompt_tokens,
            "completion": tokens.completion_tokens,
        },
        "sources": [
            {"rank": i + 1, "doc_path": path, "heading": heading}
            for i, (path, heading) in enumerate(source_entries)
        ],
        "result_count": len(results),
        "filtered_count": len(filtered),
    }
    if cfg.query_expand:
        out["expanded"] = bool(expansions)
        out["expansions"] = expansions
    return out


def similar_core(file_arg: str, cfg: Config, top_k: int = 10) -> dict:
    _require_index(cfg)

    conn = connect(cfg)

    doc_path = _resolve_doc_path(cfg, conn, file_arg)
    if not doc_path:
        conn.close()
        raise FileNotIndexedError(f"File not in index: {file_arg}")

    chunk_ids = [
        r["id"]
        for r in conn.execute(
            "SELECT c.id FROM chunks c JOIN documents d ON d.id = c.doc_id WHERE d.path = ?",
            (doc_path,),
        ).fetchall()
    ]

    if not chunk_ids:
        conn.close()
        raise FileNotIndexedError(f"No chunks found for {doc_path}.")

    embeddings = []
    for cid in chunk_ids:
        row = conn.execute(
            "SELECT embedding FROM vec_chunks WHERE chunk_id = ?", (cid,)
        ).fetchone()
        if row:
            embeddings.append(deserialize_f32(row["embedding"]))

    if not embeddings:
        conn.close()
        raise FileNotIndexedError(f"No embeddings found for {doc_path}.")

    dims = len(embeddings[0])
    avg = [sum(e[d] for e in embeddings) / len(embeddings) for d in range(dims)]

    fetch_k = top_k + len(chunk_ids) + 5
    rows = conn.execute(
        "SELECT chunk_id, distance, doc_path FROM vec_chunks "
        "WHERE embedding MATCH ? AND k = ? ORDER BY distance",
        (serialize_f32(avg), fetch_k),
    ).fetchall()

    best_per_doc: dict[str, float] = {}
    for r in rows:
        dp = r["doc_path"]
        if dp == doc_path:
            continue
        dist = r["distance"]
        if dp not in best_per_doc or dist < best_per_doc[dp]:
            best_per_doc[dp] = dist

    ranked = sorted(best_per_doc.items(), key=lambda x: x[1])[:top_k]

    titles = {}
    for r in conn.execute("SELECT path, title FROM documents").fetchall():
        titles[r["path"]] = r["title"]

    conn.close()

    return {
        "source": doc_path,
        "results": [
            {
                "rank": i + 1,
                "doc_path": dp,
                "similarity": round(1 - dist, 4),
                "title": titles.get(dp, ""),
            }
            for i, (dp, dist) in enumerate(ranked)
        ],
    }


def stats_core(cfg: Config) -> dict:
    if not cfg.db_path.exists():
        return {"error": "No index found."}

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

    documents = []
    for row in conn.execute(
        "SELECT path, title, type, chunk_count, content_hash, indexed_at FROM documents ORDER BY path"
    ):
        documents.append(
            {
                "path": row[0],
                "title": row[1],
                "type": row[2],
                "chunk_count": row[3],
                "content_hash": row[4],
                "indexed_at": row[5],
            }
        )

    conn.close()

    return {
        "db_path": str(cfg.db_path),
        "db_size_bytes": cfg.db_path.stat().st_size,
        "doc_count": doc_count,
        "chunk_count": chunk_count,
        "vec_count": vec_count,
        "fts_count": fts_count,
        "total_chars": total_chars,
        "type_counts": {r[0]: r[1] for r in type_counts},
        "documents": documents,
    }


def list_core(cfg: Config) -> dict:
    if not cfg.db_path.exists():
        return {"error": "No index found. Run 'kb index' first."}

    conn = connect(cfg)
    rows = conn.execute(
        "SELECT path, type, chunk_count, size_bytes, indexed_at "
        "FROM documents ORDER BY path"
    ).fetchall()
    conn.close()

    documents = [
        {
            "path": r["path"],
            "type": r["type"],
            "chunk_count": r["chunk_count"] or 0,
            "size_bytes": r["size_bytes"] or 0,
            "indexed_at": r["indexed_at"],
        }
        for r in rows
    ]

    return {
        "doc_count": len(documents),
        "documents": documents,
    }
