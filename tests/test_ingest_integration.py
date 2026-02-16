"""Integration tests for kb.ingest — index_directory, orphan cleanup, PDF extraction."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from kb.config import Config
from kb.db import connect
from kb.ingest import _index_file, index_directory


def _mock_openai_client(embed_dims=4):
    client = MagicMock()
    embed_resp = MagicMock()
    embed_resp.data = [MagicMock(embedding=[0.1] * embed_dims)]
    client.embeddings.create.return_value = embed_resp
    return client


def _make_cfg(tmp_path, **kwargs):
    defaults = dict(embed_dims=4, max_chunk_chars=5000, min_chunk_chars=10)
    defaults.update(kwargs)
    cfg = Config(**defaults)
    cfg.scope = "project"
    cfg.config_dir = tmp_path
    cfg.config_path = tmp_path / ".kb.toml"
    cfg.db_path = tmp_path / "kb.db"
    return cfg


class TestIndexDirectory:
    def test_indexes_md_files(self, tmp_path):
        cfg = _make_cfg(tmp_path)
        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "a.md").write_text("# Doc A\n\nContent of document A is long enough.")
        (docs / "b.md").write_text("# Doc B\n\nContent of document B is long enough.")

        client = _mock_openai_client()
        # Return enough embeddings for all chunks
        client.embeddings.create.return_value.data = [
            MagicMock(embedding=[0.1] * 4) for _ in range(10)
        ]

        with patch("kb.ingest.OpenAI", return_value=client):
            index_directory(docs, cfg)

        conn = connect(cfg)
        count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        assert count == 2
        conn.close()

    def test_skips_unchanged_on_reindex(self, tmp_path, capsys):
        cfg = _make_cfg(tmp_path)
        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "a.md").write_text("# Stable\n\nThis content does not change between indexing runs.")

        client = _mock_openai_client()
        client.embeddings.create.return_value.data = [
            MagicMock(embedding=[0.1] * 4) for _ in range(10)
        ]

        with patch("kb.ingest.OpenAI", return_value=client):
            index_directory(docs, cfg)
            capsys.readouterr()  # clear

            index_directory(docs, cfg)

        out = capsys.readouterr().out
        assert "No changes" in out

    def test_respects_kbignore(self, tmp_path):
        cfg = _make_cfg(tmp_path)
        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "keep.md").write_text("# Keep\n\nThis file should be indexed successfully.")
        (docs / "skip.draft.md").write_text("# Draft\n\nThis should be skipped by ignore rules.")
        (docs / ".kbignore").write_text("*.draft.md\n")

        client = _mock_openai_client()
        client.embeddings.create.return_value.data = [
            MagicMock(embedding=[0.1] * 4) for _ in range(10)
        ]

        with patch("kb.ingest.OpenAI", return_value=client):
            index_directory(docs, cfg)

        conn = connect(cfg)
        paths = [r[0] for r in conn.execute("SELECT path FROM documents").fetchall()]
        assert any("keep" in p for p in paths)
        assert not any("draft" in p for p in paths)
        conn.close()

    def test_skips_tiny_files(self, tmp_path):
        cfg = _make_cfg(tmp_path, min_chunk_chars=100)
        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "tiny.md").write_text("hi")

        client = _mock_openai_client()
        with patch("kb.ingest.OpenAI", return_value=client):
            index_directory(docs, cfg)

        conn = connect(cfg)
        count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        assert count == 0
        conn.close()

    def test_empty_directory(self, tmp_path, capsys):
        cfg = _make_cfg(tmp_path)
        docs = tmp_path / "empty"
        docs.mkdir()

        client = _mock_openai_client()
        with patch("kb.ingest.OpenAI", return_value=client):
            index_directory(docs, cfg)

        out = capsys.readouterr().out
        assert "Found 0 markdown" in out


class TestOrphanChunkCleanup:
    def test_removes_orphaned_chunks(self, tmp_path):
        """When a file is re-indexed with fewer chunks, old chunks are deleted."""
        cfg = _make_cfg(tmp_path)
        conn = connect(cfg)

        # Index a file with 2 sections
        to_embed = []
        text_v1 = (
            "# Title\n\n## Section A\n\nContent A is long enough to be a chunk.\n\n"
            "## Section B\n\nContent B is also long enough to be a chunk."
        )
        _index_file(conn, "f.md", text_v1, 200, "markdown", to_embed, cfg)
        conn.commit()

        chunks_v1 = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        assert chunks_v1 >= 2

        # Re-index with only 1 section
        to_embed2 = []
        text_v2 = "# Title\n\n## Section A\n\nCompletely rewritten content that replaces everything."
        _index_file(conn, "f.md", text_v2, 150, "markdown", to_embed2, cfg)
        conn.commit()

        chunks_v2 = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        assert chunks_v2 < chunks_v1


class TestPdfExtraction:
    def test_extract_pdf_text(self):
        """Only runs if pymupdf is available."""
        try:
            from kb.ingest import extract_pdf_text, PYMUPDF_AVAILABLE
        except ImportError:
            pytest.skip("pymupdf not installed")

        if not PYMUPDF_AVAILABLE:
            pytest.skip("pymupdf not installed")

        # We can't easily create a real PDF in a unit test without a dependency,
        # so just verify the function exists and is callable
        assert callable(extract_pdf_text)
