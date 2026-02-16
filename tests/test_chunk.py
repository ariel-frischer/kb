"""Tests for kb.chunk — markdown/plain text chunking, heading extraction."""

from kb.chunk import (
    _chunk_markdown_regex,
    _extract_heading_ancestry,
    _split_section,
    chunk_plain_text,
    embedding_text,
)
from kb.config import Config


class TestExtractHeadingAncestry:
    def test_single_heading(self):
        text = "# Title\nSome content"
        heading, ancestry = _extract_heading_ancestry(text)
        assert heading == "Title"
        assert ancestry == "Title"

    def test_nested_headings(self):
        text = "# Root\n## Child\n### Grandchild\nContent"
        heading, ancestry = _extract_heading_ancestry(text)
        assert heading == "Grandchild"
        assert ancestry == "Root > Child > Grandchild"

    def test_sibling_headings_pop_stack(self):
        text = "# Root\n## First\n## Second\nContent"
        heading, ancestry = _extract_heading_ancestry(text)
        assert heading == "Second"
        assert ancestry == "Root > Second"

    def test_no_headings(self):
        text = "Just plain text\nwith multiple lines"
        heading, ancestry = _extract_heading_ancestry(text)
        assert heading is None
        assert ancestry == ""

    def test_heading_level_skip(self):
        """Jumping from h1 to h3 still tracks ancestry."""
        text = "# Top\n### Deep\nContent"
        heading, ancestry = _extract_heading_ancestry(text)
        assert heading == "Deep"
        assert ancestry == "Top > Deep"

    def test_heading_resets_deeper(self):
        """Going back to a shallower heading pops deeper ones."""
        text = "# A\n## B\n### C\n## D\nContent"
        heading, ancestry = _extract_heading_ancestry(text)
        assert heading == "D"
        assert ancestry == "A > D"


class TestChunkMarkdownRegex:
    def test_splits_by_headings(self, sample_markdown):
        cfg = Config(max_chunk_chars=5000, min_chunk_chars=10)
        chunks = _chunk_markdown_regex(sample_markdown, cfg)
        assert len(chunks) > 1
        headings = [c["heading"] for c in chunks]
        assert "Installation" in headings
        assert "Usage" in headings or "Basic Commands" in headings

    def test_respects_min_chunk_chars(self):
        text = "# A\nshort\n# B\nAlso very short"
        cfg = Config(max_chunk_chars=5000, min_chunk_chars=200)
        chunks = _chunk_markdown_regex(text, cfg)
        assert all(len(c["text"]) >= 200 for c in chunks)

    def test_oversized_section_gets_split(self):
        text = "# Big Section\n\n" + "\n\n".join(
            f"Paragraph {i} with enough content to fill space." * 5 for i in range(20)
        )
        cfg = Config(max_chunk_chars=200, min_chunk_chars=10)
        chunks = _chunk_markdown_regex(text, cfg)
        assert len(chunks) > 1
        assert all(c["heading"] == "Big Section" for c in chunks)

    def test_ancestry_tracking(self):
        text = "# Top\n## Mid\nContent under mid section with enough chars to survive."
        cfg = Config(max_chunk_chars=5000, min_chunk_chars=10)
        chunks = _chunk_markdown_regex(text, cfg)
        mid_chunk = [c for c in chunks if c["heading"] == "Mid"]
        assert mid_chunk
        assert "Top > Mid" in mid_chunk[0]["heading_ancestry"]

    def test_empty_text(self):
        cfg = Config(max_chunk_chars=5000, min_chunk_chars=10)
        chunks = _chunk_markdown_regex("", cfg)
        assert chunks == []

    def test_no_headings(self):
        text = (
            "Just a paragraph with enough content to not be filtered out by min chars."
        )
        cfg = Config(max_chunk_chars=5000, min_chunk_chars=10)
        chunks = _chunk_markdown_regex(text, cfg)
        assert len(chunks) == 1
        assert chunks[0]["heading"] is None


class TestSplitSection:
    def test_small_section_not_split(self):
        cfg = Config(max_chunk_chars=500)
        chunks = _split_section("Short text.", "H", "A > H", cfg)
        assert len(chunks) == 1
        assert chunks[0]["text"] == "Short text."
        assert chunks[0]["heading"] == "H"

    def test_large_section_split_by_paragraphs(self):
        text = "\n\n".join(f"Paragraph {i} " * 10 for i in range(10))
        cfg = Config(max_chunk_chars=200)
        chunks = _split_section(text, "Heading", "Ancestry", cfg)
        assert len(chunks) > 1
        assert all(c["heading"] == "Heading" for c in chunks)
        assert all(c["heading_ancestry"] == "Ancestry" for c in chunks)


class TestChunkPlainText:
    def test_basic_plain_text(self):
        text = "\n\n".join(
            f"This is paragraph {i} with some content." for i in range(5)
        )
        cfg = Config(max_chunk_chars=5000, min_chunk_chars=10)
        chunks = chunk_plain_text(text, cfg)
        assert len(chunks) >= 1
        assert all(c["heading"] is None for c in chunks)
        assert all(c["heading_ancestry"] == "" for c in chunks)

    def test_respects_max_chunk_chars(self):
        text = "\n\n".join("Word " * 100 for _ in range(20))
        cfg = Config(max_chunk_chars=300, min_chunk_chars=10)
        chunks = chunk_plain_text(text, cfg)
        assert len(chunks) > 1

    def test_filters_small_chunks(self):
        text = "tiny\n\nAlso a very short paragraph"
        cfg = Config(max_chunk_chars=5000, min_chunk_chars=500)
        chunks = chunk_plain_text(text, cfg)
        assert chunks == []

    def test_empty_text(self):
        cfg = Config(max_chunk_chars=5000, min_chunk_chars=10)
        chunks = chunk_plain_text("", cfg)
        assert chunks == []


class TestEmbeddingText:
    def test_with_ancestry(self):
        result = embedding_text("body text", "Root > Child", "docs/file.md")
        assert result == "docs/file.md > Root > Child\n\nbody text"

    def test_without_ancestry(self):
        result = embedding_text("body text", "", "docs/file.md")
        assert result == "docs/file.md\n\nbody text"
