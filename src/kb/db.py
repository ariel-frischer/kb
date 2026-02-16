"""Database schema and connection management."""

import sqlite3
from pathlib import Path

import sqlite_vec

from .config import SCHEMA_VERSION, Config


def connect(cfg: Config) -> sqlite3.Connection:
    """Open DB, load sqlite-vec, ensure schema is current."""
    cfg.db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(cfg.db_path))
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)

    conn.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
    row = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
    current = int(row[0]) if row else 0

    if current < SCHEMA_VERSION:
        if current == 3:
            # Non-destructive migration: add tags column
            print(
                f"Schema upgrade v{current} -> v{SCHEMA_VERSION}, adding tags column..."
            )
            try:
                conn.execute("ALTER TABLE documents ADD COLUMN tags TEXT DEFAULT ''")
            except sqlite3.OperationalError:
                pass  # column already exists
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
            content_hash TEXT
        )
    """)
    conn.execute(f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
            chunk_id INTEGER PRIMARY KEY,
            embedding float[{cfg.embed_dims}],
            +chunk_text TEXT,
            +doc_path TEXT,
            +heading TEXT
        )
    """)
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS fts_chunks USING fts5(
            text,
            heading,
            content='chunks',
            content_rowid='id'
        )
    """)
    conn.commit()
    return conn


def reset(db_path: Path):
    if db_path.exists():
        db_path.unlink()
        print(f"Deleted {db_path}")
    else:
        print("No database to reset.")
