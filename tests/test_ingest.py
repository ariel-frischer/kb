"""Tests for kb.ingest — file indexing, hashing, ignore patterns."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from kb.config import Config
from kb.db import connect
from kb.ingest import (
    _index_file,
    _is_ignored,
    _load_ignore_patterns,
    _parse_frontmatter_tags,
    md5_hash,
)


class TestMd5Hash:
    def test_deterministic(self):
        assert md5_hash("hello") == md5_hash("hello")

    def test_different_inputs(self):
        assert md5_hash("a") != md5_hash("b")

    def test_returns_hex_string(self):
        result = md5_hash("test")
        assert len(result) == 32
        assert all(c in "0123456789abcdef" for c in result)


class TestLoadIgnorePatterns:
    def test_loads_from_dir(self, tmp_path):
        ignore = tmp_path / ".kbignore"
        ignore.write_text("*.draft.md\nprivate/\n# comment\n\n")
        patterns = _load_ignore_patterns(tmp_path)
        assert "*.draft.md" in patterns
        assert "private/" in patterns
        assert len(patterns) == 2  # comment and blank line excluded

    def test_loads_from_parent(self, tmp_path):
        ignore = tmp_path / ".kbignore"
        ignore.write_text("*.tmp\n")
        subdir = tmp_path / "child"
        subdir.mkdir()
        patterns = _load_ignore_patterns(subdir)
        assert "*.tmp" in patterns

    def test_no_ignore_file(self, tmp_path):
        patterns = _load_ignore_patterns(tmp_path)
        assert patterns == []


class TestIsIgnored:
    def test_matches_glob(self, tmp_path):
        (tmp_path / "draft.tmp").touch()
        assert _is_ignored(tmp_path / "draft.tmp", tmp_path, ["*.tmp"])

    def test_matches_filename(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        target = sub / "secret.md"
        target.touch()
        assert _is_ignored(target, tmp_path, ["secret.md"])

    def test_matches_directory_prefix(self, tmp_path):
        sub = tmp_path / "private" / "file.md"
        sub.parent.mkdir()
        sub.touch()
        assert _is_ignored(sub, tmp_path, ["private/"])

    def test_no_match(self, tmp_path):
        (tmp_path / "good.md").touch()
        assert not _is_ignored(tmp_path / "good.md", tmp_path, ["*.tmp"])


class TestParseFrontmatterTags:
    def test_inline_tags(self):
        text = "---\ntags: [python, tutorial]\n---\n# Hello\nBody"
        assert _parse_frontmatter_tags(text) == ["python", "tutorial"]

    def test_list_tags(self):
        text = "---\ntags:\n  - python\n  - tutorial\n---\n# Hello\nBody"
        assert _parse_frontmatter_tags(text) == ["python", "tutorial"]

    def test_no_frontmatter(self):
        assert _parse_frontmatter_tags("# Hello\nBody") == []

    def test_no_tags_field(self):
        text = "---\ntitle: Test\n---\n# Hello\nBody"
        assert _parse_frontmatter_tags(text) == []

    def test_quoted_inline_tags(self):
        text = '---\ntags: ["python", "tutorial"]\n---\n# Hello\nBody'
        assert _parse_frontmatter_tags(text) == ["python", "tutorial"]

    def test_empty_tags(self):
        text = "---\ntags: []\n---\n# Hello\nBody"
        assert _parse_frontmatter_tags(text) == []


class TestIndexFile:
    @pytest.fixture
    def index_setup(self, tmp_path):
        cfg = Config(max_chunk_chars=5000, min_chunk_chars=10)
        cfg.scope = "project"
        cfg.config_dir = tmp_path
        cfg.config_path = tmp_path / ".kb.toml"
        cfg.db_path = tmp_path / "kb.db"
        conn = connect(cfg)
        return conn, cfg

    def test_indexes_new_file(self, index_setup):
        conn, cfg = index_setup
        to_embed = []
        text = "# My Doc\n\nThis is a paragraph with enough content to be a chunk."
        did_index, reused = _index_file(
            conn, "docs/test.md", text, 100, "markdown", to_embed, cfg
        )
        conn.commit()

        assert did_index is True
        assert reused == 0
        assert len(to_embed) > 0

        doc = conn.execute(
            "SELECT * FROM documents WHERE path = 'docs/test.md'"
        ).fetchone()
        assert doc is not None
        assert doc["title"] == "My Doc"
        assert doc["type"] == "markdown"

    def test_skips_unchanged_file(self, index_setup):
        conn, cfg = index_setup
        text = "# Doc\n\nContent that is long enough to be a valid chunk in the system."
        to_embed = []
        _index_file(conn, "f.md", text, 100, "markdown", to_embed, cfg)
        conn.commit()

        to_embed2 = []
        did_index, _ = _index_file(conn, "f.md", text, 100, "markdown", to_embed2, cfg)
        assert did_index is False
        assert to_embed2 == []

    def test_reindexes_changed_file(self, index_setup):
        conn, cfg = index_setup
        to_embed = []
        _index_file(
            conn,
            "f.md",
            "# V1\n\nOriginal content that is long enough for chunking purposes.",
            100,
            "markdown",
            to_embed,
            cfg,
        )
        conn.commit()

        to_embed2 = []
        did_index, _ = _index_file(
            conn,
            "f.md",
            "# V2\n\nUpdated content that is completely different from the original.",
            120,
            "markdown",
            to_embed2,
            cfg,
        )
        assert did_index is True
        assert len(to_embed2) > 0

    def test_reuses_unchanged_chunks(self, index_setup):
        conn, cfg = index_setup
        shared_section = "## Shared\n\nParagraph that stays the same across versions of the document."
        to_embed = []
        _index_file(
            conn,
            "f.md",
            f"# Title\n\nIntro paragraph with enough content for one chunk.\n\n{shared_section}",
            100,
            "markdown",
            to_embed,
            cfg,
        )
        conn.commit()
        first_embed_count = len(to_embed)

        to_embed2 = []
        did_index, reused = _index_file(
            conn,
            "f.md",
            f"# Title\n\nDifferent intro paragraph that changes the first chunk hash.\n\n{shared_section}",
            120,
            "markdown",
            to_embed2,
            cfg,
        )
        assert did_index is True
        # The shared section chunk should be reused (not re-embedded)
        assert reused > 0
        assert len(to_embed2) < first_embed_count

    def test_empty_chunks_skipped(self, index_setup):
        conn, cfg = index_setup
        to_embed = []
        did_index, _ = _index_file(conn, "f.md", "tiny", 4, "markdown", to_embed, cfg)
        assert did_index is False

    def test_extracts_title_from_heading(self, index_setup):
        conn, cfg = index_setup
        to_embed = []
        _index_file(
            conn,
            "f.md",
            "# My Great Title\n\nBody content that has enough characters for a chunk.",
            100,
            "markdown",
            to_embed,
            cfg,
        )
        conn.commit()
        doc = conn.execute("SELECT title FROM documents WHERE path = 'f.md'").fetchone()
        assert doc["title"] == "My Great Title"

    def test_title_fallback_to_stem(self, index_setup):
        conn, cfg = index_setup
        to_embed = []
        _index_file(
            conn,
            "my-file.md",
            "No heading here, just body text that is long enough for a chunk.",
            100,
            "markdown",
            to_embed,
            cfg,
        )
        conn.commit()
        doc = conn.execute(
            "SELECT title FROM documents WHERE path = 'my-file.md'"
        ).fetchone()
        assert doc["title"] == "my-file"
