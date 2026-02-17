"""Hybrid search: vector + FTS5 + RRF fusion."""

import re
import sqlite3

from .config import Config


def fts_escape(query: str) -> str | None:
    """Convert plain text to FTS5 OR query of quoted terms."""
    words = re.findall(r"\w+", query)
    if not words:
        return None
    return " OR ".join(f'"{w}"' for w in words)


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
    vec_data: dict[int, dict] = {}
    fts_data: dict[int, float] = {}  # chunk_id -> fts_rank

    for rank, (chunk_id, distance, text, doc_path, heading) in enumerate(vec_results):
        similarity = 1.0 - distance
        scores[chunk_id] = scores.get(chunk_id, 0) + similarity / (cfg.rrf_k + rank)
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
        fts_data[chunk_id] = fts_rank

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
