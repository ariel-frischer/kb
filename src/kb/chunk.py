"""Chunking strategies for markdown and plain text."""

import re

from .config import Config

try:
    from chonkie import RecursiveChunker
    from chonkie.refinery import OverlapRefinery

    CHONKIE_AVAILABLE = True
except ImportError:
    CHONKIE_AVAILABLE = False


def _extract_heading_ancestry(text: str) -> tuple[str | None, str]:
    """Extract heading and ancestry from chunk text by parsing markdown headings."""
    heading_stack: list[tuple[int, str]] = []
    last_heading = None

    for line in text.split("\n"):
        m = re.match(r"^(#{1,6})\s+(.+?)$", line)
        if m:
            level = len(m.group(1))
            heading_text = m.group(2).strip()
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, heading_text))
            last_heading = heading_text

    ancestry = " > ".join(h for _, h in heading_stack)
    return last_heading, ancestry


def chunk_markdown(text: str, cfg: Config) -> list[dict]:
    """Split markdown by heading hierarchy with ancestry tracking.

    Uses chonkie if available, otherwise falls back to regex-based splitting.
    """
    if CHONKIE_AVAILABLE:
        return _chunk_markdown_chonkie(text, cfg)
    return _chunk_markdown_regex(text, cfg)


def _chunk_markdown_chonkie(text: str, cfg: Config) -> list[dict]:
    """Chunk markdown using chonkie's RecursiveChunker with overlap."""
    chunker = RecursiveChunker.from_recipe(
        "markdown",
        lang="en",
        chunk_size=cfg.max_chunk_chars,
        min_characters_per_chunk=cfg.min_chunk_chars,
    )

    raw_chunks = chunker.chunk(text)

    overlap_size = max(50, cfg.max_chunk_chars // 7)
    refinery = OverlapRefinery(context_size=overlap_size, method="suffix")
    refined = refinery.refine(raw_chunks)

    chunks = []
    for c in refined:
        chunk_text = c.text.strip()
        if len(chunk_text) < cfg.min_chunk_chars:
            continue
        heading, ancestry = _extract_heading_ancestry(chunk_text)
        chunks.append({"text": chunk_text, "heading": heading, "heading_ancestry": ancestry})

    return chunks


def _chunk_markdown_regex(text: str, cfg: Config) -> list[dict]:
    """Fallback: regex-based heading-aware chunking."""
    chunks = []
    lines = text.split("\n")
    heading_stack: list[tuple[int, str]] = []
    current_lines: list[str] = []
    current_heading: str | None = None

    def flush():
        nonlocal current_lines
        if not current_lines:
            return
        section_text = "\n".join(current_lines).strip()
        current_lines = []
        if not section_text:
            return
        ancestry = " > ".join(h for _, h in heading_stack)
        chunks.extend(_split_section(section_text, current_heading, ancestry, cfg))

    for line in lines:
        m = re.match(r"^(#{1,6})\s+(.+?)$", line)
        if m:
            flush()
            level = len(m.group(1))
            heading_text = m.group(2).strip()
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, heading_text))
            current_heading = heading_text
            current_lines = [line]
        else:
            current_lines.append(line)

    flush()
    return [c for c in chunks if len(c["text"]) >= cfg.min_chunk_chars]


def _split_section(text: str, heading: str | None, ancestry: str, cfg: Config) -> list[dict]:
    """Split oversized section by paragraphs."""
    base = {"heading": heading, "heading_ancestry": ancestry}

    if len(text) <= cfg.max_chunk_chars:
        return [{**base, "text": text}]

    chunks = []
    paragraphs = re.split(r"\n\n+", text)
    current = ""

    for para in paragraphs:
        if len(current) + len(para) > cfg.max_chunk_chars and current:
            chunks.append({**base, "text": current.strip()})
            current = para
        else:
            current = current + "\n\n" + para if current else para

    if current.strip():
        chunks.append({**base, "text": current.strip()})

    return chunks


def chunk_plain_text(text: str, cfg: Config) -> list[dict]:
    """Chunk plain text (from PDFs) by paragraphs with size limits."""
    if CHONKIE_AVAILABLE:
        chunker = RecursiveChunker(
            chunk_size=cfg.max_chunk_chars,
            min_characters_per_chunk=cfg.min_chunk_chars,
        )
        raw_chunks = chunker.chunk(text)
        overlap_size = max(50, cfg.max_chunk_chars // 7)
        refinery = OverlapRefinery(context_size=overlap_size, method="suffix")
        refined = refinery.refine(raw_chunks)
        return [
            {"text": c.text.strip(), "heading": None, "heading_ancestry": ""}
            for c in refined
            if len(c.text.strip()) >= cfg.min_chunk_chars
        ]

    # Fallback: paragraph-based
    paragraphs = re.split(r"\n\n+", text)
    chunks = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) > cfg.max_chunk_chars and current:
            if len(current.strip()) >= cfg.min_chunk_chars:
                chunks.append({"text": current.strip(), "heading": None, "heading_ancestry": ""})
            current = para
        else:
            current = current + "\n\n" + para if current else para

    if current.strip() and len(current.strip()) >= cfg.min_chunk_chars:
        chunks.append({"text": current.strip(), "heading": None, "heading_ancestry": ""})

    return chunks


def embedding_text(text: str, ancestry: str, source_path: str) -> str:
    """Build enriched text for embedding (prepends file path + heading ancestry)."""
    if ancestry:
        return f"{source_path} > {ancestry}\n\n{text}"
    return f"{source_path}\n\n{text}"
