"""File ingestion: indexing documents across many formats."""

import hashlib
import re
import sqlite3
import time
from collections import Counter
from fnmatch import fnmatch
from pathlib import Path

from openai import OpenAI

from .chunk import CHONKIE_AVAILABLE, chunk_markdown, chunk_plain_text, embedding_text
from .config import Config
from .db import connect
from .embed import embed_batch, serialize_f32
from .extract import extract_text, supported_extensions, unavailable_formats


def md5_hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


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


def _check_file_size(file_path: Path, cfg: Config, dir_path: Path) -> bool:
    """Return True if file is OK to index, False if too large."""
    if cfg.max_file_size_mb <= 0:
        return True
    max_bytes = cfg.max_file_size_mb * 1024 * 1024
    if file_path.stat().st_size <= max_bytes:
        return True
    rel_path = cfg.doc_path_for_db(file_path, dir_path)
    abs_path = str(file_path.resolve())
    for allowed in cfg.allowed_large_files:
        if allowed == rel_path or allowed == abs_path:
            return True
    return False


def _is_ignored(file_path: Path, dir_path: Path, patterns: list[str]) -> bool:
    """Check if a file matches any ignore pattern."""
    rel = str(file_path.relative_to(dir_path))
    for pat in patterns:
        if fnmatch(rel, pat) or fnmatch(file_path.name, pat):
            return True
        if pat.endswith("/") and rel.startswith(pat):
            return True
    return False


def _parse_frontmatter_tags(text: str) -> list[str]:
    """Extract tags from YAML frontmatter if present."""
    m = re.match(r"^---\s*\n(.+?)\n---\s*\n", text, re.DOTALL)
    if not m:
        return []
    frontmatter = m.group(1)
    # Match tags: [tag1, tag2] or tags:\n  - tag1\n  - tag2
    tm = re.search(r"^tags:\s*\[([^\]]*)\]", frontmatter, re.MULTILINE)
    if tm:
        return [t.strip().strip("\"'") for t in tm.group(1).split(",") if t.strip()]
    tags = []
    in_tags = False
    for line in frontmatter.splitlines():
        if re.match(r"^tags:\s*$", line):
            in_tags = True
            continue
        if in_tags:
            m = re.match(r"^\s+-\s+(.+)", line)
            if m:
                tags.append(m.group(1).strip().strip("\"'"))
            else:
                break
    return tags


def _index_file(
    conn: sqlite3.Connection,
    rel_path: str,
    text: str,
    file_size: int,
    doc_type: str,
    to_embed: list,
    cfg: Config,
    tags: str = "",
) -> tuple[bool, int]:
    """Index a single file's text into chunks. Returns (indexed, reused_count)."""
    file_hash = md5_hash(text)

    row = conn.execute(
        "SELECT id, content_hash FROM documents WHERE path = ?", (rel_path,)
    ).fetchone()

    if row and row["content_hash"] == file_hash:
        return False, 0

    doc_id = row["id"] if row else None

    if doc_type == "markdown":
        chunks = chunk_markdown(text, cfg)
    else:
        chunks = chunk_plain_text(text, cfg)

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
            "type=?, chunk_count=?, tags=?, indexed_at=datetime('now') WHERE id=?",
            (title, file_hash, file_size, doc_type, len(chunks), tags, doc_id),
        )
    else:
        conn.execute(
            "INSERT INTO documents (path, title, type, size_bytes, content_hash, chunk_count, tags) "
            "VALUES (?,?,?,?,?,?,?)",
            (rel_path, title, doc_type, file_size, file_hash, len(chunks), tags),
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
                (
                    doc_id,
                    i,
                    chunk["text"],
                    chunk["heading"],
                    ancestry,
                    len(chunk["text"]),
                    chunk_hash,
                ),
            )
            cid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            embed_input = embedding_text(chunk["text"], ancestry, rel_path)
            to_embed.append(
                (embed_input, cid, chunk["text"], rel_path, chunk.get("heading") or "")
            )

    orphans = set(existing.keys()) - new_hashes
    for h in orphans:
        cid = existing[h]
        conn.execute("DELETE FROM vec_chunks WHERE chunk_id = ?", (cid,))
        conn.execute("DELETE FROM chunks WHERE id = ?", (cid,))

    new_count = len(chunks) - file_reused
    reuse_note = f" ({file_reused} reused)" if file_reused else ""
    print(f"  {rel_path}: {len(chunks)} chunks, {new_count} new{reuse_note}")

    return True, file_reused


def index_directory(dir_path: Path, cfg: Config, *, no_size_limit: bool = False):
    """Index all supported document files in a directory."""
    conn = connect(cfg)
    client = OpenAI()

    ignore_patterns = _load_ignore_patterns(dir_path)
    exts = supported_extensions(include_code=cfg.index_code)

    # Collect files matching supported extensions
    all_files = sorted(
        f for f in dir_path.rglob("*") if f.is_file() and f.suffix.lower() in exts
    )
    files = [f for f in all_files if not _is_ignored(f, dir_path, ignore_patterns)]

    ignored_count = len(all_files) - len(files)
    if ignored_count:
        print(f"  Ignored {ignored_count} files via .kbignore")

    # Count by extension for summary
    ext_counts: Counter[str] = Counter()
    for f in files:
        ext_counts[f.suffix.lower()] += 1

    # Print discovery summary
    print(f"Found {len(files)} files in {dir_path}")
    if ext_counts:
        parts = [f"{cnt} {ext}" for ext, cnt in ext_counts.most_common()]
        print(f"  Types: {', '.join(parts)}")

    # Warn about unavailable optional formats
    missing = unavailable_formats()
    if missing:
        skip_counts: Counter[str] = Counter()
        for f in dir_path.rglob("*"):
            if f.is_file() and f.suffix.lower() in {ext for ext, _ in missing}:
                skip_counts[f.suffix.lower()] += 1
        for ext, pkg in missing:
            if ext in skip_counts:
                print(f"  {skip_counts[ext]} {ext} files skipped (install {pkg})")

    if CHONKIE_AVAILABLE:
        print("  (using chonkie for chunking)")

    start = time.time()
    skipped = 0
    indexed = 0
    chunks_reused = 0
    size_skipped = 0
    to_embed: list[tuple[str, int, str, str, str]] = []

    for file_path in files:
        if not no_size_limit and not _check_file_size(file_path, cfg, dir_path):
            rel = cfg.doc_path_for_db(file_path, dir_path)
            mb = file_path.stat().st_size / (1024 * 1024)
            print(
                f"  SKIP (too large): {rel} ({mb:.1f} MB > {cfg.max_file_size_mb} MB)"
            )
            size_skipped += 1
            continue

        rel_path = cfg.doc_path_for_db(file_path, dir_path)

        try:
            result = extract_text(file_path, include_code=cfg.index_code)
        except Exception as e:
            print(f"  WARN: failed to extract {rel_path}: {e}")
            continue

        if result is None:
            continue
        text, doc_type = result

        if len(text.strip()) < cfg.min_chunk_chars:
            continue

        tags = ""
        if doc_type == "markdown":
            parsed_tags = _parse_frontmatter_tags(text)
            if parsed_tags:
                tags = ",".join(parsed_tags)

        did_index, reused = _index_file(
            conn,
            rel_path,
            text,
            file_path.stat().st_size,
            doc_type,
            to_embed,
            cfg,
            tags=tags,
        )
        if did_index:
            indexed += 1
            chunks_reused += reused
        else:
            skipped += 1

    conn.commit()

    if not to_embed and indexed == 0:
        elapsed = time.time() - start
        size_note = f", {size_skipped} too large" if size_skipped else ""
        print(f"\nNo changes. ({skipped} files unchanged{size_note}, {elapsed:.1f}s)")
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
    print("\n--- Indexing complete ---")
    size_note = f", {size_skipped} too large" if size_skipped else ""
    print(f"Files: {indexed} indexed, {skipped} unchanged{size_note}")
    print(f"Chunks: {len(to_embed)} embedded, {chunks_reused} reused, {total} total")
    print(f"Time: {elapsed:.2f}s")
    print(f"DB: {cfg.db_path} ({cfg.db_path.stat().st_size / 1024:.1f} KB)")

    conn.close()
