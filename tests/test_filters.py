"""Tests for kb.filters — query filter parsing and application."""

import sqlite3

import pytest

from kb.filters import apply_filters, has_active_filters, parse_filters


class TestParseFilters:
    def test_no_filters(self):
        query, filters = parse_filters("simple query")
        assert query == "simple query"
        assert filters["file_glob"] is None
        assert filters["date_after"] is None
        assert filters["date_before"] is None
        assert filters["must_contain"] == []
        assert filters["must_not_contain"] == []

    def test_file_glob(self):
        query, filters = parse_filters("file:docs/*.md search terms")
        assert query == "search terms"
        assert filters["file_glob"] == "docs/*.md"

    def test_date_after(self):
        query, filters = parse_filters('dt>"2026-01-15" recent stuff')
        assert query == "recent stuff"
        assert filters["date_after"] == "2026-01-15"

    def test_date_before(self):
        query, filters = parse_filters('dt<"2026-02-01" old stuff')
        assert query == "old stuff"
        assert filters["date_before"] == "2026-02-01"

    def test_must_contain(self):
        query, filters = parse_filters('+"python" +"async" query here')
        assert query == "query here"
        assert filters["must_contain"] == ["python", "async"]

    def test_must_not_contain(self):
        query, filters = parse_filters('-"draft" -"wip" final results')
        assert query == "final results"
        assert filters["must_not_contain"] == ["draft", "wip"]

    def test_combined_filters(self):
        query, filters = parse_filters(
            'file:notes/*.md dt>"2026-01-01" +"important" search query'
        )
        assert query == "search query"
        assert filters["file_glob"] == "notes/*.md"
        assert filters["date_after"] == "2026-01-01"
        assert filters["must_contain"] == ["important"]

    def test_type_glob(self):
        query, filters = parse_filters("type:markdown search terms")
        assert query == "search terms"
        assert filters["type_glob"] == "markdown"

    def test_tag_single(self):
        query, filters = parse_filters("tag:python search terms")
        assert query == "search terms"
        assert filters["tags"] == ["python"]

    def test_tag_multiple(self):
        query, filters = parse_filters("tag:python tag:tutorial search terms")
        assert query == "search terms"
        assert filters["tags"] == ["python", "tutorial"]

    def test_collapses_whitespace(self):
        query, _ = parse_filters("file:x.md   lots   of   space")
        assert query == "lots of space"


class TestHasActiveFilters:
    def test_no_filters(self):
        _, filters = parse_filters("plain query")
        assert not has_active_filters(filters)

    def test_with_file_glob(self):
        _, filters = parse_filters("file:*.md query")
        assert has_active_filters(filters)

    def test_with_must_contain(self):
        _, filters = parse_filters('+"keyword" query')
        assert has_active_filters(filters)


class TestApplyFilters:
    def test_no_filters_passes_all(self, sample_chunks):
        _, filters = parse_filters("plain query")
        result = apply_filters(sample_chunks, filters, None)
        assert result == sample_chunks

    def test_file_glob_filters(self, sample_chunks):
        filters = {
            "file_glob": "docs/*",
            "date_after": None,
            "date_before": None,
            "must_contain": [],
            "must_not_contain": [],
        }
        result = apply_filters(sample_chunks, filters, None)
        assert all("docs/" in r["doc_path"] for r in result)
        assert len(result) == 2

    def test_must_contain(self, sample_chunks):
        filters = {
            "file_glob": None,
            "date_after": None,
            "date_before": None,
            "must_contain": ["install"],
            "must_not_contain": [],
        }
        result = apply_filters(sample_chunks, filters, None)
        assert len(result) == 1
        assert result[0]["chunk_id"] == 1

    def test_must_not_contain(self, sample_chunks):
        filters = {
            "file_glob": None,
            "date_after": None,
            "date_before": None,
            "must_contain": [],
            "must_not_contain": ["contributing"],
        }
        result = apply_filters(sample_chunks, filters, None)
        assert all(r["chunk_id"] != 3 for r in result)

    def test_type_glob_filters(self, tmp_path):
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        conn.execute(
            "CREATE TABLE documents (id INTEGER PRIMARY KEY, path TEXT, type TEXT)"
        )
        conn.execute("INSERT INTO documents VALUES (1, 'docs/guide.md', 'markdown')")
        conn.execute("INSERT INTO documents VALUES (2, 'docs/manual.pdf', 'pdf')")
        conn.commit()

        results = [
            {"chunk_id": 1, "doc_path": "docs/guide.md", "text": "markdown content"},
            {"chunk_id": 2, "doc_path": "docs/manual.pdf", "text": "pdf content"},
        ]
        _, filters = parse_filters("type:markdown query")
        result = apply_filters(results, filters, conn)
        assert len(result) == 1
        assert result[0]["doc_path"] == "docs/guide.md"
        conn.close()

    def test_tag_filters(self, tmp_path):
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        conn.execute(
            "CREATE TABLE documents (id INTEGER PRIMARY KEY, path TEXT, tags TEXT)"
        )
        conn.execute("INSERT INTO documents VALUES (1, 'a.md', 'python,tutorial')")
        conn.execute("INSERT INTO documents VALUES (2, 'b.md', 'rust')")
        conn.execute("INSERT INTO documents VALUES (3, 'c.md', '')")
        conn.commit()

        results = [
            {"chunk_id": 1, "doc_path": "a.md", "text": "a"},
            {"chunk_id": 2, "doc_path": "b.md", "text": "b"},
            {"chunk_id": 3, "doc_path": "c.md", "text": "c"},
        ]
        _, filters = parse_filters("tag:python query")
        result = apply_filters(results, filters, conn)
        assert len(result) == 1
        assert result[0]["doc_path"] == "a.md"
        conn.close()

    def test_tag_filters_multiple(self, tmp_path):
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        conn.execute(
            "CREATE TABLE documents (id INTEGER PRIMARY KEY, path TEXT, tags TEXT)"
        )
        conn.execute("INSERT INTO documents VALUES (1, 'a.md', 'python,tutorial')")
        conn.execute("INSERT INTO documents VALUES (2, 'b.md', 'python')")
        conn.commit()

        results = [
            {"chunk_id": 1, "doc_path": "a.md", "text": "a"},
            {"chunk_id": 2, "doc_path": "b.md", "text": "b"},
        ]
        _, filters = parse_filters("tag:python tag:tutorial query")
        result = apply_filters(results, filters, conn)
        assert len(result) == 1
        assert result[0]["doc_path"] == "a.md"
        conn.close()

    def test_date_filters(self, tmp_path):
        # Need a real DB for date filtering
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        conn.execute(
            "CREATE TABLE documents (id INTEGER PRIMARY KEY, path TEXT, indexed_at TEXT)"
        )
        conn.execute("INSERT INTO documents VALUES (1, 'old.md', '2025-12-01')")
        conn.execute("INSERT INTO documents VALUES (2, 'new.md', '2026-02-10')")
        conn.commit()

        results = [
            {"chunk_id": 1, "doc_path": "old.md", "text": "old content"},
            {"chunk_id": 2, "doc_path": "new.md", "text": "new content"},
        ]
        filters = {
            "file_glob": None,
            "date_after": "2026-01-01",
            "date_before": None,
            "must_contain": [],
            "must_not_contain": [],
        }
        result = apply_filters(results, filters, conn)
        assert len(result) == 1
        assert result[0]["doc_path"] == "new.md"
        conn.close()
