"""Hybrid search: vector + FTS5 + RRF fusion."""

import re
import sqlite3

from .config import Config


def fts_escape(query: str) -> str | None:
    """Convert plain text to FTS5 AND query with prefix matching."""
    words = re.findall(r"\w+", query)
    if not words:
        return None
    if len(words) == 1:
        return f'"{words[0]}"*'
    return " AND ".join(f'"{w}"*' for w in words)


def _rank_bonus(rank: int) -> float:
    """Positional bonus for top-ranked results."""
    if rank == 0:
        return 0.05
    if rank <= 2:
        return 0.02
    return 0.0


def rrf_fuse(
    vec_results: list[tuple],
    fts_results: list[tuple],
    top_k: int,
    cfg: Config,
) -> list[dict]:
    """Score-weighted RRF: each rank contribution is scaled by its normalized score.

    Vec: similarity / (k + rank), where similarity = 1 - distance
    FTS: norm_bm25 / (k + rank), where norm_bm25 = |score| / (1 + |score|)
    """
    scores: dict[int, float] = {}
    top_rank: dict[int, int] = {}  # chunk_id -> best rank across all lists
    vec_data: dict[int, dict] = {}
    fts_data: dict[int, float] = {}  # chunk_id -> fts_rank

    for rank, (chunk_id, distance, text, doc_path, heading) in enumerate(vec_results):
        similarity = 1.0 - distance
        scores[chunk_id] = scores.get(chunk_id, 0) + similarity / (cfg.rrf_k + rank)
        top_rank[chunk_id] = min(top_rank.get(chunk_id, rank), rank)
        vec_data[chunk_id] = {
            "distance": distance,
            "text": text,
            "doc_path": doc_path,
            "heading": heading,
        }

    for rank, (chunk_id, fts_rank) in enumerate(fts_results):
        abs_rank = abs(fts_rank)
        norm_bm25 = abs_rank / (1.0 + abs_rank)
        scores[chunk_id] = scores.get(chunk_id, 0) + norm_bm25 / (cfg.rrf_k + rank)
        top_rank[chunk_id] = min(top_rank.get(chunk_id, rank), rank)
        fts_data[chunk_id] = fts_rank

    # Apply rank bonus once per chunk based on best rank across all lists
    for chunk_id in scores:
        scores[chunk_id] += _rank_bonus(top_rank[chunk_id])

    ranked = sorted(scores.items(), key=lambda x: -x[1])[:top_k]

    results = []
    for chunk_id, rrf_score in ranked:
        data = vec_data.get(chunk_id)
        results.append(
            {
                "chunk_id": chunk_id,
                "rrf_score": rrf_score,
                "distance": data["distance"] if data else None,
                "similarity": (1 - data["distance"]) if data else None,
                "fts_rank": fts_data.get(chunk_id),
                "text": data["text"] if data else None,
                "doc_path": data["doc_path"] if data else None,
                "heading": data["heading"] if data else None,
                "in_fts": chunk_id in fts_data,
                "in_vec": data is not None,
            }
        )

    return results


def run_vec_query(conn: sqlite3.Connection, emb_bytes: bytes, k: int) -> list[tuple]:
    """Execute vec0 KNN search. Returns list of (chunk_id, distance, text, doc_path, heading)."""
    rows = conn.execute(
        "SELECT chunk_id, distance, chunk_text, doc_path, heading "
        "FROM vec_chunks WHERE embedding MATCH ? AND k = ? ORDER BY distance",
        (emb_bytes, k),
    ).fetchall()
    return [
        (r["chunk_id"], r["distance"], r["chunk_text"], r["doc_path"], r["heading"])
        for r in rows
    ]


def run_fts_query(conn: sqlite3.Connection, query_str: str, limit: int) -> list[tuple]:
    """Execute FTS5 search. Returns list of (chunk_id, fts_rank). Returns [] if no terms."""
    fts_q = fts_escape(query_str)
    if not fts_q:
        return []
    try:
        rows = conn.execute(
            "SELECT rowid, rank FROM fts_chunks WHERE fts_chunks MATCH ? ORDER BY rank LIMIT ?",
            (fts_q, limit),
        ).fetchall()
        return [(r["rowid"], r["rank"]) for r in rows]
    except sqlite3.OperationalError:
        return []


def normalize_vec_list(
    vec_results: list[tuple],
) -> list[dict]:
    """Convert vec tuples -> [{"chunk_id", "score" (=similarity), "text", "doc_path", "heading"}]."""
    return [
        {
            "chunk_id": chunk_id,
            "score": 1.0 - distance,
            "distance": distance,
            "text": text,
            "doc_path": doc_path,
            "heading": heading,
        }
        for chunk_id, distance, text, doc_path, heading in vec_results
    ]


def normalize_fts_list(fts_results: list[tuple]) -> list[dict]:
    """Convert FTS tuples -> [{"chunk_id", "score" (=norm_bm25)}]."""
    return [
        {
            "chunk_id": chunk_id,
            "score": abs(fts_rank) / (1.0 + abs(fts_rank)),
            "fts_rank": fts_rank,
        }
        for chunk_id, fts_rank in fts_results
    ]


def multi_rrf_fuse(
    scored_lists: list[list[dict]],
    weights: list[float],
    top_k: int,
    rrf_k: float = 60.0,
) -> list[dict]:
    """N-list weighted RRF with rank bonuses.

    Each scored_list is [{"chunk_id", "score", ...metadata}].
    Accumulates: weight * score / (rrf_k + rank) per chunk across all lists.
    Adds rank bonus based on best rank across all lists.

    Returns same dict format as rrf_fuse for backward compat.
    """
    scores: dict[int, float] = {}
    top_rank: dict[int, int] = {}
    vec_data: dict[int, dict] = {}
    fts_data: dict[int, float] = {}

    for list_idx, (slist, weight) in enumerate(zip(scored_lists, weights)):
        for rank, item in enumerate(slist):
            cid = item["chunk_id"]
            scores[cid] = scores.get(cid, 0) + weight * item["score"] / (rrf_k + rank)
            top_rank[cid] = min(top_rank.get(cid, rank), rank)

            # Collect metadata from vec-type lists (have "distance" key)
            if "distance" in item and cid not in vec_data:
                vec_data[cid] = {
                    "distance": item["distance"],
                    "text": item.get("text"),
                    "doc_path": item.get("doc_path"),
                    "heading": item.get("heading"),
                }
            if "fts_rank" in item:
                fts_data[cid] = item["fts_rank"]

    for cid in scores:
        scores[cid] += _rank_bonus(top_rank[cid])

    ranked = sorted(scores.items(), key=lambda x: -x[1])[:top_k]

    results = []
    for cid, rrf_score in ranked:
        data = vec_data.get(cid)
        results.append(
            {
                "chunk_id": cid,
                "rrf_score": rrf_score,
                "distance": data["distance"] if data else None,
                "similarity": (1 - data["distance"]) if data else None,
                "fts_rank": fts_data.get(cid),
                "text": data["text"] if data else None,
                "doc_path": data["doc_path"] if data else None,
                "heading": data["heading"] if data else None,
                "in_fts": cid in fts_data,
                "in_vec": data is not None,
            }
        )

    return results


def run_fts_query_filtered(
    conn: sqlite3.Connection, query_str: str, limit: int, allowed_ids: set[int]
) -> list[tuple]:
    """Execute FTS5 search restricted to specific chunk IDs (SQL-level pre-filter).

    Returns list of (chunk_id, fts_rank). Returns [] if no terms or no allowed IDs.
    """
    fts_q = fts_escape(query_str)
    if not fts_q or not allowed_ids:
        return []
    ids_csv = ",".join(str(i) for i in sorted(allowed_ids))
    try:
        rows = conn.execute(
            f"SELECT rowid, rank FROM fts_chunks "
            f"WHERE fts_chunks MATCH ? AND rowid IN ({ids_csv}) "
            f"ORDER BY rank LIMIT ?",
            (fts_q, limit),
        ).fetchall()
        return [(r["rowid"], r["rank"]) for r in rows]
    except sqlite3.OperationalError:
        return []


def run_vec_query_filtered(
    conn: sqlite3.Connection, emb_bytes: bytes, allowed_ids: set[int]
) -> list[tuple]:
    """Compute cosine distance for specific chunk IDs via vec_distance_cosine().

    SQL-level pre-filter: only computes distance for chunks in allowed_ids.
    Returns same format as run_vec_query: [(chunk_id, distance, text, doc_path, heading)].
    """
    if not allowed_ids:
        return []
    ids_csv = ",".join(str(i) for i in sorted(allowed_ids))
    rows = conn.execute(
        f"SELECT chunk_id, vec_distance_cosine(embedding, ?) as distance, "
        f"chunk_text, doc_path, heading "
        f"FROM vec_chunks WHERE chunk_id IN ({ids_csv}) "
        f"ORDER BY distance",
        (emb_bytes,),
    ).fetchall()
    return [
        (r["chunk_id"], r["distance"], r["chunk_text"], r["doc_path"], r["heading"])
        for r in rows
    ]


def fill_fts_only_results(conn: sqlite3.Connection, results: list[dict]):
    """Backfill text/metadata for results that came only from FTS."""
    for r in results:
        if r["text"] is not None:
            continue
        row = conn.execute(
            "SELECT c.text, d.path, c.heading "
            "FROM chunks c JOIN documents d ON d.id = c.doc_id WHERE c.id = ?",
            (r["chunk_id"],),
        ).fetchone()
        if row:
            r["text"] = row["text"]
            r["doc_path"] = row["path"]
            r["heading"] = row["heading"]
