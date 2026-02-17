"""Tests for kb.search — FTS escape, RRF fusion, backfill."""

import pytest

from kb.config import Config
from kb.search import fill_fts_only_results, fts_escape, rrf_fuse


class TestFtsEscape:
    def test_single_word(self):
        assert fts_escape("hello") == '"hello"'

    def test_multiple_words(self):
        result = fts_escape("hello world")
        assert result == '"hello" OR "world"'

    def test_strips_punctuation(self):
        result = fts_escape("what's the cost?")
        assert result == '"what" OR "s" OR "the" OR "cost"'

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
        # Vec: similarity(0.7) / 60 + 0.05, FTS: norm_bm25(2/3) / 60 + 0.05
        vec_contrib = 0.7 / 60.0 + 0.05
        fts_contrib = (2.0 / 3.0) / 60.0 + 0.05
        assert results[0]["rrf_score"] == pytest.approx(vec_contrib + fts_contrib)
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
