"""Database schema and connection management."""

import sqlite3
from pathlib import Path

import sqlite_vec

from .config import SCHEMA_VERSION, Config


def fts_path(doc_path: str) -> str:
    """Return last 2 path components for FTS indexing.

    Reduces IDF collapse from common prefixes (e.g. project name in every path).
    'openclaw-config/agents/AGENTS.md' -> 'agents/AGENTS.md'
    'notes/guide.md' -> 'notes/guide.md'
    'file.md' -> 'file.md'
    """
    parts = Path(doc_path).parts
    if len(parts) <= 2:
        return doc_path
    return str(Path(*parts[-2:]))


def connect(cfg: Config) -> sqlite3.Connection:
    """Open DB, load sqlite-vec, ensure schema is current."""
    cfg.db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(cfg.db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)

    conn.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
    row = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
    current = int(row[0]) if row else 0
    needs_fts_rebuild = False

    if current < SCHEMA_VERSION:
        if current == 8:
            # Vec0 used L2 distance — switch to cosine. Must drop+recreate vec_chunks.
            print(
                f"Schema upgrade v{current} -> v{SCHEMA_VERSION}, switching vec0 to cosine distance..."
            )
            conn.execute("DROP TABLE IF EXISTS vec_chunks")
            print(
                "  Dropped vec_chunks (L2). Run 'kb index' to reindex with cosine distance."
            )
        elif current == 7:
            # Non-destructive: add fts_path to chunks, rebuild FTS using truncated paths
            print(
                f"Schema upgrade v{current} -> v{SCHEMA_VERSION}, truncating FTS paths..."
            )
            for trigger in ("fts_ai", "fts_ad", "fts_au"):
                conn.execute(f"DROP TRIGGER IF EXISTS {trigger}")
            conn.execute("DROP TABLE IF EXISTS fts_chunks")
            try:
                conn.execute("ALTER TABLE chunks ADD COLUMN fts_path TEXT DEFAULT ''")
            except sqlite3.OperationalError:
                pass  # column already exists
            # Populate fts_path from doc_path (last 2 components)
            rows = conn.execute("SELECT id, doc_path FROM chunks").fetchall()
            for row in rows:
                conn.execute(
                    "UPDATE chunks SET fts_path = ? WHERE id = ?",
                    (fts_path(row["doc_path"]), row["id"]),
                )
            needs_fts_rebuild = True
        elif current == 6:
            # Non-destructive: add doc_path + fts_path to chunks, rebuild FTS
            print(
                f"Schema upgrade v{current} -> v{SCHEMA_VERSION}, adding doc_path to FTS..."
            )
            for trigger in ("fts_ai", "fts_ad", "fts_au"):
                conn.execute(f"DROP TRIGGER IF EXISTS {trigger}")
            conn.execute("DROP TABLE IF EXISTS fts_chunks")
            try:
                conn.execute("ALTER TABLE chunks ADD COLUMN doc_path TEXT DEFAULT ''")
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute("ALTER TABLE chunks ADD COLUMN fts_path TEXT DEFAULT ''")
            except sqlite3.OperationalError:
                pass
            conn.execute(
                "UPDATE chunks SET doc_path = "
                "(SELECT path FROM documents WHERE id = chunks.doc_id)"
            )
            rows = conn.execute("SELECT id, doc_path FROM chunks").fetchall()
            for row in rows:
                conn.execute(
                    "UPDATE chunks SET fts_path = ? WHERE id = ?",
                    (fts_path(row["doc_path"]), row["id"]),
                )
            needs_fts_rebuild = True
        elif current == 5:
            # Non-destructive: rebuild FTS with porter tokenizer + doc_path + fts_path
            print(
                f"Schema upgrade v{current} -> v{SCHEMA_VERSION}, rebuilding FTS with porter tokenizer..."
            )
            for trigger in ("fts_ai", "fts_ad", "fts_au"):
                conn.execute(f"DROP TRIGGER IF EXISTS {trigger}")
            conn.execute("DROP TABLE IF EXISTS fts_chunks")
            try:
                conn.execute("ALTER TABLE chunks ADD COLUMN doc_path TEXT DEFAULT ''")
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute("ALTER TABLE chunks ADD COLUMN fts_path TEXT DEFAULT ''")
            except sqlite3.OperationalError:
                pass
            conn.execute(
                "UPDATE chunks SET doc_path = "
                "(SELECT path FROM documents WHERE id = chunks.doc_id)"
            )
            rows = conn.execute("SELECT id, doc_path FROM chunks").fetchall()
            for row in rows:
                conn.execute(
                    "UPDATE chunks SET fts_path = ? WHERE id = ?",
                    (fts_path(row["doc_path"]), row["id"]),
                )
            needs_fts_rebuild = True
        elif current == 4:
            # Non-destructive: rebuild FTS with triggers + porter tokenizer + fts_path
            print(
                f"Schema upgrade v{current} -> v{SCHEMA_VERSION}, rebuilding FTS with triggers..."
            )
            for trigger in ("fts_ai", "fts_ad", "fts_au"):
                conn.execute(f"DROP TRIGGER IF EXISTS {trigger}")
            conn.execute("DROP TABLE IF EXISTS fts_chunks")
            try:
                conn.execute("ALTER TABLE chunks ADD COLUMN doc_path TEXT DEFAULT ''")
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute("ALTER TABLE chunks ADD COLUMN fts_path TEXT DEFAULT ''")
            except sqlite3.OperationalError:
                pass
            conn.execute(
                "UPDATE chunks SET doc_path = "
                "(SELECT path FROM documents WHERE id = chunks.doc_id)"
            )
            rows = conn.execute("SELECT id, doc_path FROM chunks").fetchall()
            for row in rows:
                conn.execute(
                    "UPDATE chunks SET fts_path = ? WHERE id = ?",
                    (fts_path(row["doc_path"]), row["id"]),
                )
            needs_fts_rebuild = True
        elif current == 3:
            # Non-destructive migration: add tags column + doc_path + fts_path, rebuild FTS
            print(
                f"Schema upgrade v{current} -> v{SCHEMA_VERSION}, adding tags column + FTS triggers..."
            )
            for trigger in ("fts_ai", "fts_ad", "fts_au"):
                conn.execute(f"DROP TRIGGER IF EXISTS {trigger}")
            conn.execute("DROP TABLE IF EXISTS fts_chunks")
            try:
                conn.execute("ALTER TABLE documents ADD COLUMN tags TEXT DEFAULT ''")
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute("ALTER TABLE chunks ADD COLUMN doc_path TEXT DEFAULT ''")
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute("ALTER TABLE chunks ADD COLUMN fts_path TEXT DEFAULT ''")
            except sqlite3.OperationalError:
                pass
            conn.execute(
                "UPDATE chunks SET doc_path = "
                "(SELECT path FROM documents WHERE id = chunks.doc_id)"
            )
            rows = conn.execute("SELECT id, doc_path FROM chunks").fetchall()
            for row in rows:
                conn.execute(
                    "UPDATE chunks SET fts_path = ? WHERE id = ?",
                    (fts_path(row["doc_path"]), row["id"]),
                )
            needs_fts_rebuild = True
        else:
            print(
                f"Schema upgrade v{current} -> v{SCHEMA_VERSION}, rebuilding tables..."
            )
            for table in ["vec_chunks", "fts_chunks", "chunks", "documents"]:
                conn.execute(f"DROP TABLE IF EXISTS {table}")
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
        conn.commit()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT UNIQUE NOT NULL,
            title TEXT,
            type TEXT,
            size_bytes INTEGER,
            content_hash TEXT,
            indexed_at TEXT DEFAULT (datetime('now')),
            chunk_count INTEGER DEFAULT 0,
            tags TEXT DEFAULT ''
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            chunk_index INTEGER NOT NULL,
            text TEXT NOT NULL,
            heading TEXT,
            heading_ancestry TEXT,
            char_count INTEGER,
            content_hash TEXT,
            doc_path TEXT DEFAULT '',
            fts_path TEXT DEFAULT ''
        )
    """)
    conn.execute(f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
            chunk_id INTEGER PRIMARY KEY,
            embedding float[{cfg.embed_dims}] distance_metric=cosine,
            +chunk_text TEXT,
            +doc_path TEXT,
            +heading TEXT
        )
    """)
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS fts_chunks USING fts5(
            doc_path,
            heading,
            text,
            content='chunks',
            content_rowid='id',
            tokenize='porter unicode61'
        )
    """)
    # Set weighted BM25: doc_path=10x, heading=2x, text=1x
    # This makes rank column use weighted bm25 automatically for all queries
    conn.execute("""
        INSERT INTO fts_chunks(fts_chunks, rank)
        VALUES('rank', 'bm25(10.0, 2.0, 1.0)')
    """)
    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS fts_ai AFTER INSERT ON chunks BEGIN
            INSERT INTO fts_chunks(rowid, doc_path, heading, text)
            VALUES (new.id, new.fts_path, new.heading, new.text);
        END
    """)
    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS fts_ad AFTER DELETE ON chunks BEGIN
            INSERT INTO fts_chunks(fts_chunks, rowid, doc_path, heading, text)
            VALUES ('delete', old.id, old.fts_path, old.heading, old.text);
        END
    """)
    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS fts_au AFTER UPDATE ON chunks BEGIN
            INSERT INTO fts_chunks(fts_chunks, rowid, doc_path, heading, text)
            VALUES ('delete', old.id, old.fts_path, old.heading, old.text);
            INSERT INTO fts_chunks(rowid, doc_path, heading, text)
            VALUES (new.id, new.fts_path, new.heading, new.text);
        END
    """)
    if needs_fts_rebuild:
        # Manual rebuild using fts_path (not doc_path) for the FTS doc_path column.
        # Can't use FTS5's built-in 'rebuild' because content='chunks' maps
        # FTS doc_path -> chunks.doc_path (full path), but we want truncated paths.
        conn.execute("INSERT INTO fts_chunks(fts_chunks) VALUES('delete-all')")
        conn.execute(
            "INSERT INTO fts_chunks(rowid, doc_path, heading, text) "
            "SELECT id, fts_path, heading, text FROM chunks"
        )
    conn.commit()
    return conn


def reset(db_path: Path):
    if db_path.exists():
        db_path.unlink()
        # Clean up WAL sidecar files
        for suffix in ("-shm", "-wal"):
            sidecar = db_path.parent / (db_path.name + suffix)
            if sidecar.exists():
                sidecar.unlink()
        print(f"Deleted {db_path}")
    else:
        print("No database to reset.")
