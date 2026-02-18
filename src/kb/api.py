"""Core API functions for kb. Used by both CLI and MCP server."""

import sqlite3
import time
from copy import copy
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI

from .config import GLOBAL_DATA_DIR, Config
from .db import connect
from .embed import deserialize_f32, embed_batch, serialize_f32
from .expand import expand_query
from .filters import (
    apply_filters,
    get_tagged_chunk_ids,
    has_active_filters,
    parse_filters,
    remove_tag_filter,
)
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
    run_fts_query_filtered,
    run_vec_query,
    run_vec_query_filtered,
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


def _pick_best_vec(
    conn: sqlite3.Connection,
    client: OpenAI,
    query: str,
    hyde_passage: str | None,
    retrieve_k: int,
    cfg: Config,
    tagged_ids: set[int] | None = None,
) -> tuple[list[tuple], float]:
    """Embed query (and optionally HyDE passage), run vec queries, return best results + embed_ms.

    When hyde_passage is provided, embeds both in a single batch API call,
    runs two vec queries, and returns whichever has better top-1 similarity.
    HyDE can only help, never hurt.

    When tagged_ids is provided, uses SQL-level pre-filtering via
    vec_distance_cosine() instead of KNN, ensuring tagged chunks are always found.
    """
    texts = [query]
    if hyde_passage:
        texts.append(hyde_passage)

    t0 = time.time()
    embeddings = embed_batch(client, texts, cfg, is_query=True)
    embed_ms = (time.time() - t0) * 1000

    if tagged_ids is not None:
        query_vec = run_vec_query_filtered(
            conn, serialize_f32(embeddings[0]), tagged_ids
        )
    else:
        query_vec = run_vec_query(conn, serialize_f32(embeddings[0]), retrieve_k)

    if hyde_passage and len(embeddings) > 1:
        if tagged_ids is not None:
            hyde_vec = run_vec_query_filtered(
                conn, serialize_f32(embeddings[1]), tagged_ids
            )
        else:
            hyde_vec = run_vec_query(conn, serialize_f32(embeddings[1]), retrieve_k)
        # Pick whichever has better top-1 similarity (lower distance = better)
        query_best = query_vec[0][1] if query_vec else float("inf")
        hyde_best = hyde_vec[0][1] if hyde_vec else float("inf")
        if hyde_best < query_best:
            return hyde_vec, embed_ms

    return query_vec, embed_ms


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
    if cfg.hyde_enabled:
        hyde_passage, hyde_ms = generate_hyde_passage(clean_query, client, cfg)

    has_threshold = cfg.search_threshold > 0
    retrieve_k = (top_k * 5) if has_filters else (top_k * 3)

    # Pre-compute tagged chunk IDs for pre-retrieval filtering
    tagged_chunk_ids: set[int] | None = None
    if filters.get("tags"):
        tagged_chunk_ids = get_tagged_chunk_ids(filters, conn)
        retrieve_k = max(retrieve_k, len(tagged_chunk_ids) + top_k)

    if cfg.query_expand:
        expansions, expand_ms = expand_query(client, clean_query, cfg)
        lex_exps = [e for e in expansions if e["type"] == "lex"]
        vec_exps = [e for e in expansions if e["type"] == "vec"]

        # Best-of-two vec: embed query + HyDE, pick better result set
        t0 = time.time()
        primary_vec, embed_ms = _pick_best_vec(
            conn,
            client,
            clean_query,
            hyde_passage,
            retrieve_k,
            cfg,
            tagged_ids=tagged_chunk_ids,
        )

        # Batch embed vec expansions (separate call)
        if vec_exps:
            exp_embeddings = embed_batch(
                client, [e["text"] for e in vec_exps], cfg, is_query=True
            )
            embed_ms += (time.time() - t0) * 1000 - embed_ms
            if tagged_chunk_ids is not None:
                exp_vec = [
                    run_vec_query_filtered(conn, serialize_f32(emb), tagged_chunk_ids)
                    for emb in exp_embeddings
                ]
            else:
                exp_vec = [
                    run_vec_query(conn, serialize_f32(emb), retrieve_k)
                    for emb in exp_embeddings
                ]
        else:
            exp_vec = []

        t0 = time.time()
        vec_ms = (time.time() - t0) * 1000

        t0 = time.time()
        if tagged_chunk_ids is not None:
            primary_fts = run_fts_query_filtered(
                conn, clean_query, retrieve_k, tagged_chunk_ids
            )
            exp_fts = [
                run_fts_query_filtered(conn, e["text"], retrieve_k, tagged_chunk_ids)
                for e in lex_exps
            ]
        else:
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
        fused_count = len(results)

        vec_count = len(primary_vec)
        fts_count = len(primary_fts)
    else:
        t0 = time.time()
        vec_results, embed_ms = _pick_best_vec(
            conn,
            client,
            clean_query,
            hyde_passage,
            retrieve_k,
            cfg,
            tagged_ids=tagged_chunk_ids,
        )
        vec_ms = (time.time() - t0) * 1000 - embed_ms

        t0 = time.time()
        if tagged_chunk_ids is not None:
            fts_results = run_fts_query_filtered(
                conn, clean_query, retrieve_k, tagged_chunk_ids
            )
        else:
            fts_results = run_fts_query(conn, clean_query, retrieve_k)
        fts_ms = (time.time() - t0) * 1000

        fuse_k = retrieve_k if has_filters else top_k
        results = rrf_fuse(vec_results, fts_results, fuse_k, cfg)
        fill_fts_only_results(conn, results)
        fused_count = len(results)

        vec_count = len(vec_results)
        fts_count = len(fts_results)

    # Skip tag in post-filters since it was handled pre-fusion
    post_filters = (
        remove_tag_filter(filters) if tagged_chunk_ids is not None else filters
    )
    if has_active_filters(post_filters):
        results = apply_filters(results, post_filters, conn)

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
            "fused": fused_count,
            **({"after_filters": len(results)} if has_filters else {}),
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

    # BM25 shortcut — deduplicate by document before checking gap
    bm25_shortcut = False
    fts_q = fts_escape(clean_question)
    if fts_q and not has_filters:
        try:
            probe = conn.execute(
                "SELECT rowid, rank FROM fts_chunks WHERE fts_chunks MATCH ? ORDER BY rank LIMIT 20",
                (fts_q,),
            ).fetchall()
            if probe:
                # Group by doc_path, keep best norm per doc
                best_per_doc: dict[str, float] = {}
                for row in probe:
                    chunk_row = conn.execute(
                        "SELECT doc_path FROM chunks WHERE id = ?", (row["rowid"],)
                    ).fetchone()
                    doc_path = chunk_row["doc_path"] if chunk_row else str(row["rowid"])
                    abs_rank = abs(row["rank"])
                    norm = abs_rank / (1.0 + abs_rank)
                    if doc_path not in best_per_doc or norm > best_per_doc[doc_path]:
                        best_per_doc[doc_path] = norm
                doc_norms = sorted(best_per_doc.values(), reverse=True)
                top_norm = doc_norms[0]
                second_norm = doc_norms[1] if len(doc_norms) > 1 else 0.0
                if (
                    top_norm >= cfg.bm25_shortcut_min
                    and (top_norm - second_norm) >= cfg.bm25_shortcut_gap
                ):
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
        hyde_passage = None
        if cfg.hyde_enabled:
            hyde_passage, hyde_ms = generate_hyde_passage(clean_question, client, cfg)

        retrieve_k = max(cfg.rerank_fetch_k, top_k * 3)

        # Pre-compute tagged chunk IDs for pre-retrieval filtering
        tagged_chunk_ids_ask: set[int] | None = None
        if filters.get("tags"):
            tagged_chunk_ids_ask = get_tagged_chunk_ids(filters, conn)
            retrieve_k = max(retrieve_k, len(tagged_chunk_ids_ask) + top_k)

        if cfg.query_expand:
            expansions, expand_ms = expand_query(client, clean_question, cfg)
            lex_exps = [e for e in expansions if e["type"] == "lex"]
            vec_exps = [e for e in expansions if e["type"] == "vec"]

            # Best-of-two vec: embed query + HyDE, pick better result set
            t0 = time.time()
            primary_vec, embed_ms = _pick_best_vec(
                conn,
                client,
                clean_question,
                hyde_passage,
                retrieve_k,
                cfg,
                tagged_ids=tagged_chunk_ids_ask,
            )

            # Batch embed vec expansions
            if vec_exps:
                exp_embeddings = embed_batch(
                    client, [e["text"] for e in vec_exps], cfg, is_query=True
                )
                if tagged_chunk_ids_ask is not None:
                    exp_vec = [
                        run_vec_query_filtered(
                            conn, serialize_f32(emb), tagged_chunk_ids_ask
                        )
                        for emb in exp_embeddings
                    ]
                else:
                    exp_vec = [
                        run_vec_query(conn, serialize_f32(emb), retrieve_k)
                        for emb in exp_embeddings
                    ]
            else:
                exp_vec = []

            if tagged_chunk_ids_ask is not None:
                primary_fts = run_fts_query_filtered(
                    conn, clean_question, retrieve_k, tagged_chunk_ids_ask
                )
                exp_fts = [
                    run_fts_query_filtered(
                        conn, e["text"], retrieve_k, tagged_chunk_ids_ask
                    )
                    for e in lex_exps
                ]
            else:
                primary_fts = run_fts_query(conn, clean_question, retrieve_k)
                exp_fts = [run_fts_query(conn, e["text"], retrieve_k) for e in lex_exps]
            search_ms = (time.time() - t0) * 1000 - embed_ms

            all_lists = [
                normalize_vec_list(primary_vec),
                normalize_fts_list(primary_fts),
            ]
            all_lists += [normalize_vec_list(r) for r in exp_vec]
            all_lists += [normalize_fts_list(r) for r in exp_fts]
            weights = [2.0, 2.0] + [1.0] * (len(all_lists) - 2)

            results = multi_rrf_fuse(all_lists, weights, cfg.rerank_fetch_k, cfg.rrf_k)
            fill_fts_only_results(conn, results)
            fused_count = len(results)
        else:
            t0 = time.time()
            vec_results, embed_ms = _pick_best_vec(
                conn,
                client,
                clean_question,
                hyde_passage,
                retrieve_k,
                cfg,
                tagged_ids=tagged_chunk_ids_ask,
            )

            if tagged_chunk_ids_ask is not None:
                fts_results_ask = run_fts_query_filtered(
                    conn, clean_question, retrieve_k, tagged_chunk_ids_ask
                )
            else:
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
            search_ms = (time.time() - t0) * 1000 - embed_ms

            results = rrf_fuse(vec_results, fts_results_ask, cfg.rerank_fetch_k, cfg)
            fill_fts_only_results(conn, results)
            fused_count = len(results)

        # Skip tag in post-filters since it was handled pre-fusion
        post_filters_ask = (
            remove_tag_filter(filters) if tagged_chunk_ids_ask is not None else filters
        )
        if has_active_filters(post_filters_ask):
            results = apply_filters(results, post_filters_ask, conn)

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
        "result_count": fused_count if not bm25_shortcut else len(results),
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


# ---------------------------------------------------------------------------
# Feedback
# ---------------------------------------------------------------------------

FEEDBACK_PATH = GLOBAL_DATA_DIR / "feedback.yml"
_VALID_SEVERITIES = ("bug", "suggestion", "note")


def _yaml_escape(s: str) -> str:
    """Escape a string for safe inclusion in a YAML value (block scalar not needed)."""
    if not s:
        return '""'
    # Use double-quoted scalar if the string contains special chars
    if any(
        c in s
        for c in (
            ":",
            "#",
            "'",
            '"',
            "\n",
            "{",
            "}",
            "[",
            "]",
            ",",
            "&",
            "*",
            "!",
            "|",
            ">",
            "%",
            "@",
            "`",
        )
    ):
        escaped = s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
        return f'"{escaped}"'
    return s


def _feedback_entry_to_yaml(entry: dict) -> str:
    """Serialize a feedback entry dict to a YAML string (manual, no deps)."""
    lines = ["- timestamp: " + entry["timestamp"]]
    lines.append("  kb_version: " + _yaml_escape(entry["kb_version"]))
    lines.append("  message: " + _yaml_escape(entry["message"]))
    lines.append("  severity: " + entry["severity"])
    for key in ("tool", "context", "agent_id", "error_trace"):
        if entry.get(key):
            lines.append(f"  {key}: " + _yaml_escape(entry[key]))
    return "\n".join(lines) + "\n"


def _parse_feedback_yaml(text: str) -> list[dict]:
    """Parse the feedback YAML file into a list of entry dicts (minimal parser)."""
    entries: list[dict] = []
    current: dict | None = None
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if line.startswith("- "):
            if current is not None:
                entries.append(current)
            current = {}
            line = "  " + line[2:]  # normalize to indented key
        if current is None:
            continue
        if line.startswith("  ") and ":" in line:
            key, _, val = line.strip().partition(":")
            val = val.strip()
            # Unquote double-quoted values
            if val.startswith('"') and val.endswith('"') and len(val) >= 2:
                val = (
                    val[1:-1]
                    .replace("\\n", "\n")
                    .replace('\\"', '"')
                    .replace("\\\\", "\\")
                )
            elif val == '""':
                val = ""
            current[key] = val
    if current is not None:
        entries.append(current)
    return entries


def feedback_core(
    message: str,
    tool: str = "",
    severity: str = "note",
    context: str = "",
    agent_id: str = "",
    error_trace: str = "",
) -> dict:
    """Submit a feedback entry. Returns the entry dict."""
    if not message or not message.strip():
        raise KBError("Feedback message cannot be empty.")
    if severity not in _VALID_SEVERITIES:
        raise KBError(
            f"Invalid severity '{severity}'. Must be one of: {', '.join(_VALID_SEVERITIES)}"
        )

    from importlib.metadata import version

    entry = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "kb_version": version("kb"),
        "message": message.strip(),
        "severity": severity,
        "tool": tool,
        "context": context,
        "agent_id": agent_id,
        "error_trace": error_trace,
    }

    FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FEEDBACK_PATH, "a") as f:
        f.write(_feedback_entry_to_yaml(entry))

    return entry


def list_feedback_core() -> dict:
    """List all feedback entries. Returns dict with entries list."""
    if not FEEDBACK_PATH.exists():
        return {"count": 0, "entries": []}

    text = FEEDBACK_PATH.read_text()
    if not text.strip():
        return {"count": 0, "entries": []}

    entries = _parse_feedback_yaml(text)
    return {"count": len(entries), "entries": entries}
