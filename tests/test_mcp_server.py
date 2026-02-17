"""Tests for the MCP server tools."""

from unittest.mock import MagicMock, patch

import pytest

from kb.config import Config
from kb.db import connect
from kb.embed import serialize_f32
from kb.mcp_server import kb_ask, kb_fts, kb_list, kb_search, kb_similar, kb_status


@pytest.fixture
def mock_config(tmp_path):
    """Config + DB with test data."""
    cfg = Config(embed_dims=4)
    cfg.scope = "project"
    cfg.config_dir = tmp_path
    cfg.config_path = tmp_path / ".kb.toml"
    cfg.db_path = tmp_path / "kb.db"

    conn = connect(cfg)
    conn.execute(
        "INSERT INTO documents (path, title, type, size_bytes, content_hash, chunk_count) "
        "VALUES ('docs/guide.md', 'Guide', 'markdown', 500, 'abc123', 1)"
    )
    doc_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.execute(
        "INSERT INTO chunks (doc_id, chunk_index, text, heading, char_count, content_hash) "
        "VALUES (?, 0, 'Install kb with pip install kb.', 'Setup', 31, 'h0')",
        (doc_id,),
    )
    chunk_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    emb = [0.1] * 4
    conn.execute(
        "INSERT INTO vec_chunks (chunk_id, embedding, chunk_text, doc_path, heading) "
        "VALUES (?, ?, ?, ?, ?)",
        (
            chunk_id,
            serialize_f32(emb),
            "Install kb with pip install kb.",
            "docs/guide.md",
            "Setup",
        ),
    )
    conn.execute("INSERT INTO fts_chunks(fts_chunks) VALUES('rebuild')")
    conn.commit()
    conn.close()
    return cfg


def _mock_openai_client():
    client = MagicMock()
    embed_resp = MagicMock()
    embed_resp.data = [MagicMock(embedding=[0.1] * 4)]
    client.embeddings.create.return_value = embed_resp
    chat_resp = MagicMock()
    chat_resp.choices = [MagicMock()]
    chat_resp.choices[0].message.content = "Answer from sources [1]."
    chat_resp.usage = MagicMock(prompt_tokens=100, completion_tokens=30)
    client.chat.completions.create.return_value = chat_resp
    return client


class TestKbSearch:
    def test_returns_results(self, mock_config):
        client = _mock_openai_client()
        with (
            patch("kb.api.OpenAI", return_value=client),
            patch("kb.mcp_server._get_config", return_value=mock_config),
        ):
            result = kb_search("install")
        assert "results" in result
        assert result["query"] == "install"

    def test_no_index_returns_error(self, tmp_path):
        cfg = Config()
        cfg.db_path = tmp_path / "nonexistent.db"
        with patch("kb.mcp_server._get_config", return_value=cfg):
            result = kb_search("query")
        assert "error" in result


class TestKbFts:
    def test_returns_results(self, mock_config):
        with patch("kb.mcp_server._get_config", return_value=mock_config):
            result = kb_fts("install")
        assert "results" in result
        assert len(result["results"]) >= 1

    def test_no_terms_returns_error(self, mock_config):
        with patch("kb.mcp_server._get_config", return_value=mock_config):
            # Query with only filter syntax, no searchable terms
            result = kb_fts("file:*.md")
        assert "error" in result


class TestKbAsk:
    def test_returns_answer(self, mock_config):
        client = _mock_openai_client()
        with (
            patch("kb.api.OpenAI", return_value=client),
            patch("kb.mcp_server._get_config", return_value=mock_config),
        ):
            result = kb_ask("How do I install?")
        assert "answer" in result


class TestKbSimilar:
    def test_file_not_indexed(self, mock_config):
        with patch("kb.mcp_server._get_config", return_value=mock_config):
            result = kb_similar("nonexistent.md")
        assert "error" in result

    def test_returns_results(self, mock_config):
        with patch("kb.mcp_server._get_config", return_value=mock_config):
            result = kb_similar("docs/guide.md")
        assert "results" in result


class TestKbStatus:
    def test_returns_stats(self, mock_config):
        with patch("kb.mcp_server._get_config", return_value=mock_config):
            result = kb_status()
        assert result["doc_count"] == 1
        assert result["chunk_count"] == 1


class TestKbList:
    def test_returns_documents(self, mock_config):
        with patch("kb.mcp_server._get_config", return_value=mock_config):
            result = kb_list()
        assert result["doc_count"] == 1
        assert result["documents"][0]["path"] == "docs/guide.md"
