"""Tests for the heavier CLI commands: stats, index, search, ask."""

from unittest.mock import MagicMock, patch

import pytest

from kb.cli import (
    cmd_ask,
    cmd_completion,
    cmd_index,
    cmd_list,
    cmd_search,
    cmd_similar,
    cmd_stats,
    cmd_tag,
    cmd_tags,
    cmd_untag,
)
from kb.config import Config
from kb.db import connect
from kb.embed import serialize_f32


@pytest.fixture
def populated_db(tmp_path):
    """Config + DB with a document, chunks, vec_chunks, and fts_chunks populated."""
    cfg = Config(embed_dims=4)
    cfg.scope = "project"
    cfg.config_dir = tmp_path
    cfg.config_path = tmp_path / ".kb.toml"
    cfg.db_path = tmp_path / "kb.db"

    conn = connect(cfg)

    conn.execute(
        "INSERT INTO documents (path, title, type, size_bytes, content_hash, chunk_count) "
        "VALUES ('docs/guide.md', 'Guide', 'markdown', 500, 'abc123', 2)"
    )
    doc_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    for i, (text, heading) in enumerate(
        [
            ("Install kb with pip install kb from PyPI.", "Installation"),
            ("Search your knowledge base using kb search query.", "Usage"),
        ]
    ):
        conn.execute(
            "INSERT INTO chunks (doc_id, chunk_index, text, heading, heading_ancestry, char_count, content_hash) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (doc_id, i, text, heading, f"Guide > {heading}", len(text), f"hash{i}"),
        )
        chunk_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

        emb = [0.1 * (i + 1)] * 4
        conn.execute(
            "INSERT INTO vec_chunks (chunk_id, embedding, chunk_text, doc_path, heading) "
            "VALUES (?, ?, ?, ?, ?)",
            (chunk_id, serialize_f32(emb), text, "docs/guide.md", heading),
        )

    conn.execute("INSERT INTO fts_chunks(fts_chunks) VALUES('rebuild')")
    conn.commit()
    conn.close()

    return cfg


def _mock_openai_client(embed_dims=4):
    """Build a mock OpenAI client that handles embeddings and chat."""
    client = MagicMock()

    # Embeddings
    embed_resp = MagicMock()
    embed_resp.data = [MagicMock(embedding=[0.1] * embed_dims)]
    client.embeddings.create.return_value = embed_resp

    # Chat completions
    chat_resp = MagicMock()
    chat_resp.choices = [MagicMock()]
    chat_resp.choices[0].message.content = "Here is the answer based on sources [1]."
    chat_resp.usage = MagicMock(prompt_tokens=200, completion_tokens=50)
    client.chat.completions.create.return_value = chat_resp

    return client


class TestCmdStats:
    def test_no_db(self, tmp_path, capsys):
        cfg = Config()
        cfg.db_path = tmp_path / "nonexistent.db"
        cmd_stats(cfg)
        assert "No index" in capsys.readouterr().out

    def test_with_data(self, populated_db, capsys):
        cmd_stats(populated_db)
        out = capsys.readouterr().out
        assert "Documents: 1" in out
        assert "Chunks: 2" in out
        assert "Vectors: 2" in out
        assert "docs/guide.md" in out

    def test_shows_capabilities(self, populated_db, capsys):
        cmd_stats(populated_db)
        out = capsys.readouterr().out
        assert "chonkie" in out
        assert "PDF" in out or "pdf" in out.lower()
        assert "rerank" in out.lower()


class TestCmdIndex:
    def test_indexes_markdown_files(self, tmp_path):
        cfg = Config(embed_dims=4, max_chunk_chars=5000, min_chunk_chars=10)
        cfg.scope = "project"
        cfg.config_dir = tmp_path
        cfg.config_path = tmp_path / ".kb.toml"
        cfg.db_path = tmp_path / "kb.db"

        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "file.md").write_text(
            "# Hello\n\nThis is a test document with enough content."
        )

        mock_client = _mock_openai_client(embed_dims=4)
        mock_client.embeddings.create.return_value.data = [
            MagicMock(embedding=[0.1] * 4)
        ]

        with patch("kb.ingest.OpenAI", return_value=mock_client):
            cmd_index(cfg, [str(docs)])

        conn = connect(cfg)
        doc_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        assert doc_count == 1
        conn.close()

    def test_no_sources_exits(self, tmp_path):
        cfg = Config()
        cfg.scope = "project"
        cfg.config_dir = tmp_path
        cfg.sources = []
        with pytest.raises(SystemExit):
            cmd_index(cfg, [])

    def test_nonexistent_dir_exits(self, tmp_path):
        cfg = Config()
        cfg.scope = "project"
        cfg.config_dir = tmp_path
        with pytest.raises(SystemExit):
            cmd_index(cfg, [str(tmp_path / "nope")])


class TestCmdSearch:
    def test_no_db_exits(self, tmp_path):
        cfg = Config()
        cfg.db_path = tmp_path / "nonexistent.db"
        with pytest.raises(SystemExit):
            cmd_search("query", cfg)

    def test_basic_search(self, populated_db, capsys):
        client = _mock_openai_client(embed_dims=4)
        with patch("kb.cli.OpenAI", return_value=client):
            cmd_search("install", populated_db, top_k=5)

        out = capsys.readouterr().out
        assert "install" in out.lower()
        assert "Embed:" in out
        assert "Vec:" in out

    def test_search_with_filter(self, populated_db, capsys):
        client = _mock_openai_client(embed_dims=4)
        with patch("kb.cli.OpenAI", return_value=client):
            cmd_search('file:docs/*.md +"install" search query', populated_db, top_k=5)

        out = capsys.readouterr().out
        assert "Filters:" in out

    def test_search_top_k(self, populated_db, capsys):
        client = _mock_openai_client(embed_dims=4)
        with patch("kb.cli.OpenAI", return_value=client):
            cmd_search("query", populated_db, top_k=1)

        out = capsys.readouterr().out
        # Should have at most 1 result block
        assert out.count("--- [") <= 1


class TestCmdAsk:
    def test_no_db_exits(self, tmp_path):
        cfg = Config()
        cfg.db_path = tmp_path / "nonexistent.db"
        with pytest.raises(SystemExit):
            cmd_ask("question", cfg)

    def test_basic_ask(self, populated_db, capsys):
        client = _mock_openai_client(embed_dims=4)
        with patch("kb.cli.OpenAI", return_value=client):
            cmd_ask("How do I install?", populated_db, top_k=5)

        out = capsys.readouterr().out
        assert "How do I install?" in out
        assert "Sources" in out

    def test_ask_calls_rerank_when_enough_results(self, populated_db, capsys):
        populated_db.rerank_top_k = 1  # force rerank to trigger
        client = _mock_openai_client(embed_dims=4)

        # Rerank response
        rerank_resp = MagicMock()
        rerank_resp.choices = [MagicMock()]
        rerank_resp.choices[0].message.content = "1, 2"
        rerank_resp.usage = MagicMock(prompt_tokens=100, completion_tokens=10)

        # chat.completions.create called twice: rerank then answer
        client.chat.completions.create.side_effect = [
            rerank_resp,
            client.chat.completions.create.return_value,
        ]

        with patch("kb.cli.OpenAI", return_value=client):
            cmd_ask("question", populated_db, top_k=5)

        assert client.chat.completions.create.call_count == 2

    def test_ask_no_results_above_threshold(self, tmp_path, capsys):
        """When all results have similarity below threshold, show 'no relevant documents'."""
        # Build a DB where vec results have high distance (low similarity)
        cfg = Config(embed_dims=4, min_similarity=0.99)
        cfg.scope = "project"
        cfg.config_dir = tmp_path
        cfg.config_path = tmp_path / ".kb.toml"
        cfg.db_path = tmp_path / "kb.db"

        conn = connect(cfg)
        conn.execute(
            "INSERT INTO documents (path, title, type, size_bytes, content_hash, chunk_count) "
            "VALUES ('d.md', 'D', 'markdown', 100, 'h', 1)"
        )
        doc_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            "INSERT INTO chunks (doc_id, chunk_index, text, heading, char_count) "
            "VALUES (?, 0, 'some text', 'H', 9)",
            (doc_id,),
        )
        chunk_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        # High distance = low similarity (0.05)
        emb = [0.9] * 4
        conn.execute(
            "INSERT INTO vec_chunks (chunk_id, embedding, chunk_text, doc_path, heading) "
            "VALUES (?, ?, ?, ?, ?)",
            (chunk_id, serialize_f32(emb), "some text", "d.md", "H"),
        )
        conn.execute("INSERT INTO fts_chunks(fts_chunks) VALUES('rebuild')")
        conn.commit()
        conn.close()

        # Mock embeddings to return vector far from the stored one
        client = _mock_openai_client(embed_dims=4)
        client.embeddings.create.return_value.data = [MagicMock(embedding=[0.1] * 4)]

        with patch("kb.cli.OpenAI", return_value=client):
            cmd_ask("question", cfg, top_k=5)

        out = capsys.readouterr().out
        assert "No relevant documents" in out

    def test_ask_with_filters(self, populated_db, capsys):
        client = _mock_openai_client(embed_dims=4)
        with patch("kb.cli.OpenAI", return_value=client):
            cmd_ask('file:docs/*.md +"install" how to install?', populated_db)

        out = capsys.readouterr().out
        assert "Filters:" in out

    def test_ask_sources_show_numbered_citations(self, populated_db, capsys):
        client = _mock_openai_client(embed_dims=4)
        with patch("kb.cli.OpenAI", return_value=client):
            cmd_ask("How do I install?", populated_db, top_k=5)

        out = capsys.readouterr().out
        assert "--- Sources ---" in out
        assert "[1]" in out
        assert "docs/guide.md" in out

    def test_ask_sources_show_heading_ancestry(self, populated_db, capsys):
        """Sources footer should display heading ancestry breadcrumbs."""
        client = _mock_openai_client(embed_dims=4)
        with patch("kb.cli.OpenAI", return_value=client):
            cmd_ask("How do I install?", populated_db, top_k=5)

        out = capsys.readouterr().out
        # The populated_db fixture stores heading_ancestry as "Guide > Installation" etc.
        assert "Guide >" in out

    def test_ask_sources_dedup_by_heading(self, tmp_path, capsys):
        """Different sections of the same file should appear as separate source entries."""
        cfg = Config(embed_dims=4)
        cfg.scope = "project"
        cfg.config_dir = tmp_path
        cfg.config_path = tmp_path / ".kb.toml"
        cfg.db_path = tmp_path / "kb.db"

        conn = connect(cfg)
        conn.execute(
            "INSERT INTO documents (path, title, type, size_bytes, content_hash, chunk_count) "
            "VALUES ('doc.md', 'Doc', 'markdown', 500, 'abc', 2)"
        )
        doc_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

        for i, (text, heading, ancestry) in enumerate(
            [
                ("First section content with enough text.", "Setup", "Doc > Setup"),
                ("Second section content with enough text.", "Usage", "Doc > Usage"),
            ]
        ):
            conn.execute(
                "INSERT INTO chunks (doc_id, chunk_index, text, heading, heading_ancestry, char_count, content_hash) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (doc_id, i, text, heading, ancestry, len(text), f"h{i}"),
            )
            chunk_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            emb = [0.1 * (i + 1)] * 4
            conn.execute(
                "INSERT INTO vec_chunks (chunk_id, embedding, chunk_text, doc_path, heading) "
                "VALUES (?, ?, ?, ?, ?)",
                (chunk_id, serialize_f32(emb), text, "doc.md", heading),
            )

        conn.execute("INSERT INTO fts_chunks(fts_chunks) VALUES('rebuild')")
        conn.commit()
        conn.close()

        client = _mock_openai_client(embed_dims=4)
        with patch("kb.cli.OpenAI", return_value=client):
            cmd_ask("question", cfg, top_k=5)

        out = capsys.readouterr().out
        # Both sections should appear as separate citations
        assert "[1]" in out
        assert "[2]" in out
        assert "Doc > Setup" in out
        assert "Doc > Usage" in out


class TestCmdList:
    def test_no_db(self, tmp_path, capsys):
        cfg = Config()
        cfg.db_path = tmp_path / "nonexistent.db"
        cmd_list(cfg)
        assert "No index" in capsys.readouterr().out

    def test_empty_db(self, tmp_path, capsys):
        cfg = Config(embed_dims=4)
        cfg.scope = "project"
        cfg.config_dir = tmp_path
        cfg.config_path = tmp_path / ".kb.toml"
        cfg.db_path = tmp_path / "kb.db"
        conn = connect(cfg)
        conn.close()

        cmd_list(cfg)
        assert "No documents" in capsys.readouterr().out

    def test_lists_documents_summary(self, populated_db, capsys):
        cmd_list(populated_db)
        out = capsys.readouterr().out
        assert "1 documents indexed" in out
        assert "markdown" in out
        assert "2 chunks" in out
        assert "kb list --full" in out

    def test_lists_documents_full(self, populated_db, capsys):
        cmd_list(populated_db, full=True)
        out = capsys.readouterr().out
        assert "1 documents indexed" in out
        assert "docs/guide.md" in out
        assert "markdown" in out
        assert "2 chunks" in out

    def test_formats_size_kb(self, tmp_path, capsys):
        cfg = Config(embed_dims=4)
        cfg.scope = "project"
        cfg.config_dir = tmp_path
        cfg.config_path = tmp_path / ".kb.toml"
        cfg.db_path = tmp_path / "kb.db"

        conn = connect(cfg)
        conn.execute(
            "INSERT INTO documents (path, title, type, size_bytes, content_hash, chunk_count) "
            "VALUES ('f.md', 'F', 'markdown', 12400, 'h', 3)"
        )
        conn.commit()
        conn.close()

        cmd_list(cfg, full=True)
        out = capsys.readouterr().out
        assert "12.4 KB" in out

    def test_formats_size_mb(self, tmp_path, capsys):
        cfg = Config(embed_dims=4)
        cfg.scope = "project"
        cfg.config_dir = tmp_path
        cfg.config_path = tmp_path / ".kb.toml"
        cfg.db_path = tmp_path / "kb.db"

        conn = connect(cfg)
        conn.execute(
            "INSERT INTO documents (path, title, type, size_bytes, content_hash, chunk_count) "
            "VALUES ('big.pdf', 'Big', 'pdf', 2500000, 'h', 10)"
        )
        conn.commit()
        conn.close()

        cmd_list(cfg, full=True)
        out = capsys.readouterr().out
        assert "2.5 MB" in out


class TestCmdCompletion:
    def test_zsh(self, capsys):
        cmd_completion("zsh")
        out = capsys.readouterr().out
        assert "#compdef kb" in out
        assert "_kb" in out
        assert "init" in out
        assert "list" in out

    def test_bash(self, capsys):
        cmd_completion("bash")
        out = capsys.readouterr().out
        assert "complete -F _kb kb" in out
        assert "COMPREPLY" in out
        assert "compgen" in out
        assert "--project" in out

    def test_fish(self, capsys):
        cmd_completion("fish")
        out = capsys.readouterr().out
        assert "__fish_use_subcommand" in out
        assert "init" in out
        assert "--project" in out
        assert "zsh bash fish" in out

    def test_unsupported_shell(self):
        with pytest.raises(SystemExit):
            cmd_completion("powershell")


@pytest.fixture
def two_doc_db(tmp_path):
    """Config + DB with two documents for similarity testing."""
    cfg = Config(embed_dims=4)
    cfg.scope = "project"
    cfg.config_dir = tmp_path
    cfg.config_path = tmp_path / ".kb.toml"
    cfg.db_path = tmp_path / "kb.db"

    conn = connect(cfg)

    for doc_i, (path, title) in enumerate(
        [
            ("docs/guide.md", "Guide"),
            ("docs/tutorial.md", "Tutorial"),
        ]
    ):
        conn.execute(
            "INSERT INTO documents (path, title, type, size_bytes, content_hash, chunk_count) "
            "VALUES (?, ?, 'markdown', 500, ?, 1)",
            (path, title, f"hash{doc_i}"),
        )
        doc_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

        emb = [0.1 * (doc_i + 1)] * 4
        conn.execute(
            "INSERT INTO chunks (doc_id, chunk_index, text, heading, char_count, content_hash) "
            "VALUES (?, 0, ?, 'Section', 50, ?)",
            (doc_id, f"Content of {title}", f"chash{doc_i}"),
        )
        chunk_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            "INSERT INTO vec_chunks (chunk_id, embedding, chunk_text, doc_path, heading) "
            "VALUES (?, ?, ?, ?, ?)",
            (chunk_id, serialize_f32(emb), f"Content of {title}", path, "Section"),
        )

    conn.execute("INSERT INTO fts_chunks(fts_chunks) VALUES('rebuild')")
    conn.commit()
    conn.close()
    return cfg


class TestCmdSimilar:
    def test_no_db_exits(self, tmp_path):
        cfg = Config()
        cfg.db_path = tmp_path / "nonexistent.db"
        with pytest.raises(SystemExit):
            cmd_similar("file.md", cfg)

    def test_file_not_in_index(self, two_doc_db, capsys):
        with pytest.raises(SystemExit):
            cmd_similar("nonexistent.md", two_doc_db)
        assert "not in index" in capsys.readouterr().out

    def test_finds_similar(self, two_doc_db, capsys):
        cmd_similar("docs/guide.md", two_doc_db, top_k=5)
        out = capsys.readouterr().out
        assert "docs/tutorial.md" in out
        assert "sim:" in out

    def test_excludes_source_document(self, two_doc_db, capsys):
        cmd_similar("docs/guide.md", two_doc_db, top_k=5)
        out = capsys.readouterr().out
        # Source doc should not appear in results
        assert out.count("docs/guide.md") == 1  # only in the header line


class TestCmdTag:
    def test_adds_tags(self, populated_db, capsys):
        cmd_tag(populated_db, "docs/guide.md", ["python", "tutorial"])
        out = capsys.readouterr().out
        assert "python" in out
        assert "tutorial" in out

    def test_adds_to_existing_tags(self, populated_db, capsys):
        cmd_tag(populated_db, "docs/guide.md", ["python"])
        cmd_tag(populated_db, "docs/guide.md", ["tutorial"])
        out = capsys.readouterr().out
        assert "python" in out
        assert "tutorial" in out

    def test_file_not_in_index(self, populated_db, capsys):
        with pytest.raises(SystemExit):
            cmd_tag(populated_db, "nonexistent.md", ["tag1"])

    def test_no_duplicates(self, populated_db, capsys):
        cmd_tag(populated_db, "docs/guide.md", ["python"])
        capsys.readouterr()
        cmd_tag(populated_db, "docs/guide.md", ["python"])
        out = capsys.readouterr().out
        assert out.count("python") == 1


class TestCmdUntag:
    def test_removes_tags(self, populated_db, capsys):
        cmd_tag(populated_db, "docs/guide.md", ["python", "tutorial"])
        capsys.readouterr()
        cmd_untag(populated_db, "docs/guide.md", ["python"])
        out = capsys.readouterr().out
        assert "tutorial" in out
        assert "python" not in out

    def test_remove_all_tags(self, populated_db, capsys):
        cmd_tag(populated_db, "docs/guide.md", ["python"])
        capsys.readouterr()
        cmd_untag(populated_db, "docs/guide.md", ["python"])
        out = capsys.readouterr().out
        assert "All tags removed" in out

    def test_file_not_in_index(self, populated_db, capsys):
        with pytest.raises(SystemExit):
            cmd_untag(populated_db, "nonexistent.md", ["tag1"])


class TestCmdTags:
    def test_no_tags(self, populated_db, capsys):
        cmd_tags(populated_db)
        assert "No tagged" in capsys.readouterr().out

    def test_shows_tags(self, populated_db, capsys):
        cmd_tag(populated_db, "docs/guide.md", ["python", "tutorial"])
        capsys.readouterr()
        cmd_tags(populated_db)
        out = capsys.readouterr().out
        assert "python" in out
        assert "tutorial" in out
        assert "1 doc" in out
