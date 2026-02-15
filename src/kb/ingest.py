"""File ingestion: indexing markdown and PDF files."""

import hashlib
import re
import sqlite3
import time
from fnmatch import fnmatch
from pathlib import Path

from openai import OpenAI

from .chunk import chunk_markdown, chunk_plain_text, embedding_text, CHONKIE_AVAILABLE
from .config import Config
from .db import connect
from .embed import embed_batch, serialize_f32

try:
    import pymupdf

    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False


def md5_hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


def extract_pdf_text(pdf_path: Path) -> str:
    """Extract text from a PDF file using pymupdf."""
    doc = pymupdf.open(str(pdf_path))
    pages = []
    for page in doc:
        text = page.get_text()
        if text.strip():
            pages.append(text)
    doc.close()
    return "\n\n".join(pages)


def _load_ignore_patterns(dir_path: Path) -> list[str]:
    """Load ignore patterns from .kbignore files."""
    patterns = []
    for kbignore in [dir_path / ".kbignore", dir_path.parent / ".kbignore"]:
        if kbignore.is_file():
            for line in kbignore.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    patterns.append(line)
            print(f"  Loaded {len(patterns)} ignore patterns from {kbignore}")
            break
    return patterns


def _is_ignored(file_path: Path, dir_path: Path, patterns: list[str]) -> bool:
    """Check if a file matches any ignore pattern."""
    rel = str(file_path.relative_to(dir_path))
    for pat in patterns:
        if fnmatch(rel, pat) or fnmatch(file_path.name, pat):
            return True
        if pat.endswith("/") and rel.startswith(pat):
            return True
    return False


def _index_file(
    conn: sqlite3.Connection,
    rel_path: str,
    text: str,
    file_size: int,
    doc_type: str,
    to_embed: list,
    cfg: Config,
) -> tuple[bool, int]:
    """Index a single file's text into chunks. Returns (indexed, reused_count)."""
    file_hash = md5_hash(text)

    row = conn.execute(
        "SELECT id, content_hash FROM documents WHERE path = ?", (rel_path,)
    ).fetchone()

    if row and row["content_hash"] == file_hash:
        return False, 0

    doc_id = row["id"] if row else None

    if doc_type == "pdf":
        chunks = chunk_plain_text(text, cfg)
    else:
        chunks = chunk_markdown(text, cfg)

    if not chunks:
        return False, 0

    title_match = re.match(r"^#\s+(.+?)$", text, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else Path(rel_path).stem

    existing: dict[str, int] = {}
    if doc_id:
        for r in conn.execute(
            "SELECT id, content_hash FROM chunks WHERE doc_id = ?", (doc_id,)
        ):
            if r["content_hash"]:
                existing[r["content_hash"]] = r["id"]

    if doc_id:
        conn.execute(
            "UPDATE documents SET title=?, content_hash=?, size_bytes=?, "
            "type=?, chunk_count=?, indexed_at=datetime('now') WHERE id=?",
            (title, file_hash, file_size, doc_type, len(chunks), doc_id),
        )
    else:
        conn.execute(
            "INSERT INTO documents (path, title, type, size_bytes, content_hash, chunk_count) "
            "VALUES (?,?,?,?,?,?)",
            (rel_path, title, doc_type, file_size, file_hash, len(chunks)),
        )
        doc_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    new_hashes: set[str] = set()
    file_reused = 0

    for i, chunk in enumerate(chunks):
        chunk_hash = md5_hash(chunk["text"])
        new_hashes.add(chunk_hash)
        ancestry = chunk.get("heading_ancestry", "")

        if chunk_hash in existing:
            cid = existing[chunk_hash]
            conn.execute(
                "UPDATE chunks SET chunk_index=?, heading=?, heading_ancestry=? WHERE id=?",
                (i, chunk["heading"], ancestry, cid),
            )
            file_reused += 1
        else:
            conn.execute(
                "INSERT INTO chunks "
                "(doc_id, chunk_index, text, heading, heading_ancestry, char_count, content_hash) "
                "VALUES (?,?,?,?,?,?,?)",
                (doc_id, i, chunk["text"], chunk["heading"], ancestry, len(chunk["text"]), chunk_hash),
            )
            cid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            embed_input = embedding_text(chunk["text"], ancestry, rel_path)
            to_embed.append((embed_input, cid, chunk["text"], rel_path, chunk.get("heading") or ""))

    orphans = set(existing.keys()) - new_hashes
    for h in orphans:
        cid = existing[h]
        conn.execute("DELETE FROM vec_chunks WHERE chunk_id = ?", (cid,))
        conn.execute("DELETE FROM chunks WHERE id = ?", (cid,))

    new_count = len(chunks) - file_reused
    reuse_note = f" ({file_reused} reused)" if file_reused else ""
    print(f"  {rel_path}: {len(chunks)} chunks, {new_count} new{reuse_note}")

    return True, file_reused


def index_directory(dir_path: Path, cfg: Config):
    """Index all markdown and PDF files in a directory."""
    conn = connect(cfg)
    client = OpenAI()

    ignore_patterns = _load_ignore_patterns(dir_path)

    all_md = sorted(dir_path.rglob("*.md"))
    all_pdf = sorted(dir_path.rglob("*.pdf")) if PYMUPDF_AVAILABLE else []

    md_files = [f for f in all_md if not _is_ignored(f, dir_path, ignore_patterns)]
    pdf_files = [f for f in all_pdf if not _is_ignored(f, dir_path, ignore_patterns)]

    ignored_count = (len(all_md) - len(md_files)) + (len(all_pdf) - len(pdf_files))
    if ignored_count:
        print(f"  Ignored {ignored_count} files via .kbignore")

    print(f"Found {len(md_files)} markdown files", end="")
    if pdf_files:
        print(f" + {len(pdf_files)} PDF files", end="")
    if not PYMUPDF_AVAILABLE:
        pdf_count = len(sorted(dir_path.rglob("*.pdf")))
        if pdf_count:
            print(f" ({pdf_count} PDFs skipped - install pymupdf)", end="")
    print(f" in {dir_path}")

    if CHONKIE_AVAILABLE:
        print("  (using chonkie for chunking)")

    start = time.time()
    skipped = 0
    indexed = 0
    chunks_reused = 0
    to_embed: list[tuple[str, int, str, str, str]] = []

    # Determine base path for relative paths
    # Use config_dir if available, otherwise dir_path's grandparent (legacy compat)
    base_path = cfg.config_dir if cfg.config_dir else dir_path.parent.parent

    for md_file in md_files:
        try:
            rel_path = str(md_file.relative_to(base_path))
        except ValueError:
            rel_path = str(md_file)
        text = md_file.read_text(errors="replace")
        if len(text.strip()) < cfg.min_chunk_chars:
            continue

        did_index, reused = _index_file(conn, rel_path, text, md_file.stat().st_size, "markdown", to_embed, cfg)
        if did_index:
            indexed += 1
            chunks_reused += reused
        else:
            skipped += 1

    for pdf_file in pdf_files:
        try:
            rel_path = str(pdf_file.relative_to(base_path))
        except ValueError:
            rel_path = str(pdf_file)
        try:
            text = extract_pdf_text(pdf_file)
        except Exception as e:
            print(f"  WARN: failed to extract {rel_path}: {e}")
            continue

        if len(text.strip()) < cfg.min_chunk_chars:
            continue

        did_index, reused = _index_file(conn, rel_path, text, pdf_file.stat().st_size, "pdf", to_embed, cfg)
        if did_index:
            indexed += 1
            chunks_reused += reused
        else:
            skipped += 1

    conn.commit()

    if not to_embed and indexed == 0:
        elapsed = time.time() - start
        print(f"\nNo changes. ({skipped} files unchanged, {elapsed:.1f}s)")
        conn.close()
        return

    if to_embed:
        print(f"\nEmbedding {len(to_embed)} chunks...")
        batch_size = 100
        t0 = time.time()

        for i in range(0, len(to_embed), batch_size):
            batch = to_embed[i : i + batch_size]
            texts = [b[0] for b in batch]
            embeddings = embed_batch(client, texts, cfg)

            for (_, cid, raw_text, doc_path, heading), emb in zip(batch, embeddings):
                conn.execute(
                    "INSERT INTO vec_chunks (chunk_id, embedding, chunk_text, doc_path, heading) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (cid, serialize_f32(emb), raw_text, doc_path, heading),
                )

            done = min(i + batch_size, len(to_embed))
            print(f"  {done}/{len(to_embed)}")

        print(f"  Embedding time: {time.time() - t0:.2f}s")

    conn.execute("INSERT INTO fts_chunks(fts_chunks) VALUES('rebuild')")
    conn.commit()

    total = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    elapsed = time.time() - start
    print(f"\n--- Indexing complete ---")
    print(f"Files: {indexed} indexed, {skipped} skipped")
    print(f"Chunks: {len(to_embed)} embedded, {chunks_reused} reused, {total} total")
    print(f"Time: {elapsed:.2f}s")
    print(f"DB: {cfg.db_path} ({cfg.db_path.stat().st_size / 1024:.1f} KB)")

    conn.close()
