"""Tests for kb.search — FTS escape, RRF fusion, multi-RRF fusion, backfill, SQL-level filtering."""

import struct

import pytest

from kb.config import Config
from kb.search import (
    fill_fts_only_results,
    fts_escape,
    multi_rrf_fuse,
    normalize_fts_list,
    normalize_vec_list,
    rrf_fuse,
    run_fts_query_filtered,
    run_vec_query_filtered,
)


@pytest.fixture
def db_conn_4d(tmp_path):
    """Live sqlite connection with 4-dim vec0 schema for vec tests."""
    from kb.db import connect

    cfg = Config(embed_dims=4)
    cfg.scope = "project"
    cfg.config_dir = tmp_path
    cfg.config_path = tmp_path / ".kb.toml"
    cfg.db_path = tmp_path / "kb.db"
    conn = connect(cfg)
    yield conn
    conn.close()


class TestFtsEscape:
    def test_single_word(self):
        assert fts_escape("hello") == '"hello"*'

    def test_multiple_words(self):
        result = fts_escape("hello world")
        assert result == '"hello"* AND "world"*'

    def test_strips_punctuation(self):
        result = fts_escape("what's the cost?")
        assert result == '"what"* AND "s"* AND "the"* AND "cost"*'

    def test_empty_string(self):
        assert fts_escape("") is None

    def test_only_punctuation(self):
        assert fts_escape("!@#$%") is None


class TestRrfFuse:
    def test_vec_only(self):
        cfg = Config(rrf_k=60.0)
        vec = [(10, 0.2, "text a", "a.md", "H1")]
        fts = []
        results = rrf_fuse(vec, fts, top_k=5, cfg=cfg)
        assert len(results) == 1
        assert results[0]["chunk_id"] == 10
        assert results[0]["in_vec"] is True
        assert results[0]["in_fts"] is False
        assert results[0]["similarity"] == pytest.approx(0.8)
        # Score-weighted: similarity / (k + rank) + rank_bonus = 0.8 / 60 + 0.05
        assert results[0]["rrf_score"] == pytest.approx(0.8 / 60.0 + 0.05)

    def test_fts_only(self):
        cfg = Config(rrf_k=60.0)
        vec = []
        fts = [(20, -1.5)]
        results = rrf_fuse(vec, fts, top_k=5, cfg=cfg)
        assert len(results) == 1
        assert results[0]["chunk_id"] == 20
        assert results[0]["in_fts"] is True
        assert results[0]["in_vec"] is False
        assert results[0]["text"] is None  # no vec data
        assert results[0]["fts_rank"] == -1.5
        # norm_bm25 = 1.5 / (1 + 1.5) = 0.6, + rank_bonus(0) = 0.05
        assert results[0]["rrf_score"] == pytest.approx(0.6 / 60.0 + 0.05)

    def test_overlap_boosts_score(self):
        cfg = Config(rrf_k=60.0)
        vec = [(1, 0.3, "text", "a.md", "H")]
        fts = [(1, -2.0)]
        results = rrf_fuse(vec, fts, top_k=5, cfg=cfg)
        assert len(results) == 1
        # Vec: similarity(0.7) / 60, FTS: norm_bm25(2/3) / 60, + single rank bonus(0)
        vec_contrib = 0.7 / 60.0
        fts_contrib = (2.0 / 3.0) / 60.0
        assert results[0]["rrf_score"] == pytest.approx(
            vec_contrib + fts_contrib + 0.05
        )
        assert results[0]["in_vec"] is True
        assert results[0]["in_fts"] is True

    def test_ranking_order(self):
        cfg = Config(rrf_k=60.0)
        vec = [
            (1, 0.1, "best", "a.md", "H1"),
            (2, 0.5, "mid", "b.md", "H2"),
            (3, 0.9, "worst", "c.md", "H3"),
        ]
        fts = [(3, -3.0), (2, -2.0), (1, -1.0)]  # reversed ranking
        results = rrf_fuse(vec, fts, top_k=3, cfg=cfg)
        # All three appear in both, but scores differ by rank
        ids = [r["chunk_id"] for r in results]
        assert len(ids) == 3

    def test_top_k_limits_results(self):
        cfg = Config(rrf_k=60.0)
        vec = [(i, 0.1 * i, f"text{i}", "f.md", "H") for i in range(10)]
        fts = []
        results = rrf_fuse(vec, fts, top_k=3, cfg=cfg)
        assert len(results) == 3

    def test_similarity_calculation(self):
        cfg = Config(rrf_k=60.0)
        vec = [(1, 0.25, "text", "a.md", "H")]
        results = rrf_fuse(vec, [], top_k=5, cfg=cfg)
        assert results[0]["distance"] == 0.25
        assert results[0]["similarity"] == pytest.approx(0.75)

    def test_bm25_normalization(self):
        """High BM25 scores produce higher fusion contributions than low ones."""
        cfg = Config(rrf_k=60.0)
        # Two FTS-only results at same rank position but different BM25 scores
        # Test separately to isolate normalization effect
        fts_high = [(1, -10.0)]  # high BM25
        fts_low = [(2, -0.1)]  # low BM25
        results_high = rrf_fuse([], fts_high, top_k=5, cfg=cfg)
        results_low = rrf_fuse([], fts_low, top_k=5, cfg=cfg)
        assert results_high[0]["rrf_score"] > results_low[0]["rrf_score"]
        # Verify normalization bounds: norm_bm25 is always in (0, 1)
        # High: 10/(1+10) = 0.909..., Low: 0.1/(1+0.1) = 0.0909...
        expected_high = (10.0 / 11.0) / 60.0 + 0.05
        expected_low = (0.1 / 1.1) / 60.0 + 0.05
        assert results_high[0]["rrf_score"] == pytest.approx(expected_high)
        assert results_low[0]["rrf_score"] == pytest.approx(expected_low)

    def test_rank_bonus_decreases(self):
        """Rank 0 bonus > rank 1 bonus > rank 3 bonus (zero)."""
        from kb.search import _rank_bonus

        assert _rank_bonus(0) == 0.05
        assert _rank_bonus(1) == 0.02
        assert _rank_bonus(2) == 0.02
        assert _rank_bonus(3) == 0.0
        assert _rank_bonus(0) > _rank_bonus(1) > _rank_bonus(3)


class TestRunFtsQueryFiltered:
    """Test SQL-level FTS pre-filtering by chunk IDs."""

    def _insert_doc_and_chunks(self, conn, texts):
        """Insert a doc with chunks; return list of chunk IDs."""
        conn.execute(
            "INSERT INTO documents (path, title, type, size_bytes, content_hash) "
            "VALUES ('test.md', 'Test', 'markdown', 100, 'abc')"
        )
        doc_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        chunk_ids = []
        for i, text in enumerate(texts):
            conn.execute(
                "INSERT INTO chunks (doc_id, doc_path, fts_path, chunk_index, text, heading, char_count) "
                "VALUES (?, 'test.md', 'test.md', ?, ?, '', ?)",
                (doc_id, i, text, len(text)),
            )
            chunk_ids.append(conn.execute("SELECT last_insert_rowid()").fetchone()[0])
        conn.commit()
        return chunk_ids

    def test_returns_only_allowed_ids(self, db_conn):
        ids = self._insert_doc_and_chunks(
            db_conn, ["alpha bravo charlie", "alpha delta echo", "foxtrot golf hotel"]
        )
        # Only allow first two chunks; query matches all three via "alpha" only in first two
        result = run_fts_query_filtered(db_conn, "alpha", 10, {ids[0], ids[1]})
        returned_ids = {r[0] for r in result}
        assert returned_ids == {ids[0], ids[1]}

    def test_excludes_non_allowed_ids(self, db_conn):
        ids = self._insert_doc_and_chunks(
            db_conn, ["alpha bravo", "alpha charlie", "alpha delta"]
        )
        # Only allow last chunk
        result = run_fts_query_filtered(db_conn, "alpha", 10, {ids[2]})
        assert len(result) == 1
        assert result[0][0] == ids[2]

    def test_empty_allowed_ids(self, db_conn):
        self._insert_doc_and_chunks(db_conn, ["alpha bravo"])
        assert run_fts_query_filtered(db_conn, "alpha", 10, set()) == []

    def test_no_match_in_allowed(self, db_conn):
        ids = self._insert_doc_and_chunks(db_conn, ["alpha bravo", "charlie delta"])
        # Allow only chunk that doesn't match query
        result = run_fts_query_filtered(db_conn, "alpha", 10, {ids[1]})
        assert result == []


class TestRunVecQueryFiltered:
    """Test SQL-level vec pre-filtering by chunk IDs via vec_distance_cosine."""

    def _insert_vec_chunks(self, conn, embeddings, dims=4):
        """Insert chunks + vec_chunks; return list of chunk IDs."""
        conn.execute(
            "INSERT INTO documents (path, title, type, size_bytes, content_hash) "
            "VALUES ('test.md', 'Test', 'markdown', 100, 'abc')"
        )
        doc_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        chunk_ids = []
        for i, emb in enumerate(embeddings):
            conn.execute(
                "INSERT INTO chunks (doc_id, doc_path, fts_path, chunk_index, text, heading, char_count) "
                "VALUES (?, 'test.md', 'test.md', ?, ?, '', 10)",
                (doc_id, i, f"text_{i}"),
            )
            cid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            emb_bytes = struct.pack(f"{dims}f", *emb)
            conn.execute(
                "INSERT INTO vec_chunks (chunk_id, embedding, chunk_text, doc_path, heading) "
                "VALUES (?, ?, ?, 'test.md', '')",
                (cid, emb_bytes, f"text_{i}"),
            )
            chunk_ids.append(cid)
        conn.commit()
        return chunk_ids

    def test_returns_only_allowed_ids(self, db_conn_4d):
        embs = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.7, 0.7, 0.0, 0.0],
        ]
        ids = self._insert_vec_chunks(db_conn_4d, embs)
        query_emb = struct.pack("4f", 1.0, 0.0, 0.0, 0.0)
        # Only allow first and third chunks
        result = run_vec_query_filtered(db_conn_4d, query_emb, {ids[0], ids[2]})
        returned_ids = {r[0] for r in result}
        assert returned_ids == {ids[0], ids[2]}
        # First chunk (exact match) should have distance ~0
        assert result[0][0] == ids[0]
        assert result[0][1] == pytest.approx(0.0, abs=1e-6)

    def test_empty_allowed_ids(self, db_conn_4d):
        embs = [[1.0, 0.0, 0.0, 0.0]]
        self._insert_vec_chunks(db_conn_4d, embs)
        query_emb = struct.pack("4f", 1.0, 0.0, 0.0, 0.0)
        assert run_vec_query_filtered(db_conn_4d, query_emb, set()) == []

    def test_results_sorted_by_distance(self, db_conn_4d):
        embs = [
            [0.0, 1.0, 0.0, 0.0],  # orthogonal to query
            [1.0, 0.0, 0.0, 0.0],  # exact match
            [0.7, 0.7, 0.0, 0.0],  # partial match
        ]
        ids = self._insert_vec_chunks(db_conn_4d, embs)
        query_emb = struct.pack("4f", 1.0, 0.0, 0.0, 0.0)
        result = run_vec_query_filtered(db_conn_4d, query_emb, set(ids))
        # Should be sorted: exact match first, partial second, orthogonal last
        assert result[0][0] == ids[1]  # exact match, distance ~0
        assert result[1][0] == ids[2]  # partial match
        assert result[2][0] == ids[0]  # orthogonal, distance ~1


class TestFillFtsOnlyResults:
    def test_backfills_missing_text(self, db_conn):
        # Insert a doc + chunk into the DB
        db_conn.execute(
            "INSERT INTO documents (path, title, type, size_bytes, content_hash) "
            "VALUES ('test.md', 'Test', 'markdown', 100, 'abc')"
        )
        doc_id = db_conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        db_conn.execute(
            "INSERT INTO chunks (doc_id, chunk_index, text, heading, char_count) "
            "VALUES (?, 0, 'The actual text', 'Section A', 15)",
            (doc_id,),
        )
        chunk_id = db_conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        db_conn.commit()

        results = [
            {"chunk_id": chunk_id, "text": None, "doc_path": None, "heading": None}
        ]
        fill_fts_only_results(db_conn, results)
        assert results[0]["text"] == "The actual text"
        assert results[0]["doc_path"] == "test.md"
        assert results[0]["heading"] == "Section A"

    def test_skips_already_filled(self, db_conn):
        results = [
            {
                "chunk_id": 999,
                "text": "already here",
                "doc_path": "x.md",
                "heading": "H",
            }
        ]
        fill_fts_only_results(db_conn, results)
        assert results[0]["text"] == "already here"


class TestNormalizeHelpers:
    def test_normalize_vec_list(self):
        vec = [(10, 0.2, "text a", "a.md", "H1"), (20, 0.5, "text b", "b.md", "H2")]
        result = normalize_vec_list(vec)
        assert len(result) == 2
        assert result[0]["chunk_id"] == 10
        assert result[0]["score"] == pytest.approx(0.8)
        assert result[0]["distance"] == 0.2
        assert result[0]["text"] == "text a"
        assert result[1]["score"] == pytest.approx(0.5)

    def test_normalize_fts_list(self):
        fts = [(10, -1.5), (20, -3.0)]
        result = normalize_fts_list(fts)
        assert len(result) == 2
        assert result[0]["chunk_id"] == 10
        assert result[0]["score"] == pytest.approx(1.5 / 2.5)
        assert result[0]["fts_rank"] == -1.5
        assert result[1]["score"] == pytest.approx(3.0 / 4.0)


class TestMultiRrfFuse:
    def test_single_list(self):
        items = [
            {
                "chunk_id": 10,
                "score": 0.8,
                "distance": 0.2,
                "text": "a",
                "doc_path": "a.md",
                "heading": "H",
            },
        ]
        results = multi_rrf_fuse([items], [1.0], top_k=5, rrf_k=60.0)
        assert len(results) == 1
        assert results[0]["chunk_id"] == 10
        # score = 1.0 * 0.8 / (60 + 0) + rank_bonus(0) = 0.8/60 + 0.05
        assert results[0]["rrf_score"] == pytest.approx(0.8 / 60.0 + 0.05)
        assert results[0]["in_vec"] is True

    def test_two_lists_with_weights(self):
        vec_list = [
            {
                "chunk_id": 1,
                "score": 0.9,
                "distance": 0.1,
                "text": "t",
                "doc_path": "a.md",
                "heading": "H",
            },
        ]
        fts_list = [
            {"chunk_id": 2, "score": 0.7, "fts_rank": -2.0},
        ]
        results = multi_rrf_fuse([vec_list, fts_list], [2.0, 1.0], top_k=5, rrf_k=60.0)
        assert len(results) == 2
        # chunk 1: 2.0 * 0.9 / 60 + 0.05 = 0.03 + 0.05
        # chunk 2: 1.0 * 0.7 / 60 + 0.05
        scores = {r["chunk_id"]: r["rrf_score"] for r in results}
        assert scores[1] == pytest.approx(2.0 * 0.9 / 60.0 + 0.05)
        assert scores[2] == pytest.approx(1.0 * 0.7 / 60.0 + 0.05)

    def test_overlap_across_lists(self):
        list1 = [
            {
                "chunk_id": 1,
                "score": 0.8,
                "distance": 0.2,
                "text": "t",
                "doc_path": "a.md",
                "heading": "H",
            },
        ]
        list2 = [
            {"chunk_id": 1, "score": 0.6, "fts_rank": -1.5},
        ]
        results = multi_rrf_fuse([list1, list2], [2.0, 2.0], top_k=5, rrf_k=60.0)
        assert len(results) == 1
        # Both contributions at rank 0: 2.0*0.8/60 + 2.0*0.6/60 + bonus(0)
        expected = 2.0 * 0.8 / 60.0 + 2.0 * 0.6 / 60.0 + 0.05
        assert results[0]["rrf_score"] == pytest.approx(expected)
        assert results[0]["in_vec"] is True
        assert results[0]["in_fts"] is True

    def test_rank_bonus_applied(self):
        # chunk at rank 0 should have higher bonus than chunk at rank 3
        items = [
            {
                "chunk_id": i,
                "score": 0.5,
                "distance": 0.5,
                "text": "t",
                "doc_path": "a.md",
                "heading": "H",
            }
            for i in range(5)
        ]
        results = multi_rrf_fuse([items], [1.0], top_k=5, rrf_k=60.0)
        # rank 0 chunk gets +0.05, rank 1 gets +0.02, rank 3 gets 0
        assert results[0]["rrf_score"] > results[2]["rrf_score"]
        assert results[2]["rrf_score"] > results[3]["rrf_score"]

    def test_top_k_limits(self):
        items = [
            {
                "chunk_id": i,
                "score": 0.5,
                "distance": 0.5,
                "text": "t",
                "doc_path": "a.md",
                "heading": "H",
            }
            for i in range(10)
        ]
        results = multi_rrf_fuse([items], [1.0], top_k=3, rrf_k=60.0)
        assert len(results) == 3

    def test_backward_compat_format(self):
        vec_list = [
            {
                "chunk_id": 1,
                "score": 0.8,
                "distance": 0.2,
                "text": "txt",
                "doc_path": "a.md",
                "heading": "H1",
            },
        ]
        fts_list = [
            {"chunk_id": 2, "score": 0.6, "fts_rank": -1.5},
        ]
        results = multi_rrf_fuse([vec_list, fts_list], [1.0, 1.0], top_k=5, rrf_k=60.0)
        for r in results:
            assert "chunk_id" in r
            assert "rrf_score" in r
            assert "distance" in r
            assert "similarity" in r
            assert "fts_rank" in r
            assert "text" in r
            assert "doc_path" in r
            assert "heading" in r
            assert "in_fts" in r
            assert "in_vec" in r

        # Vec-sourced chunk
        vec_r = next(r for r in results if r["chunk_id"] == 1)
        assert vec_r["distance"] == 0.2
        assert vec_r["similarity"] == pytest.approx(0.8)
        assert vec_r["in_vec"] is True
        assert vec_r["text"] == "txt"

        # FTS-sourced chunk
        fts_r = next(r for r in results if r["chunk_id"] == 2)
        assert fts_r["distance"] is None
        assert fts_r["similarity"] is None
        assert fts_r["in_fts"] is True
        assert fts_r["in_vec"] is False
