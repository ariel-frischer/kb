"""Tests for kb.db — schema creation, migration, reset."""

import sqlite3


from kb.config import SCHEMA_VERSION, Config
from kb.db import connect, reset


class TestConnect:
    def test_creates_db_and_tables(self, tmp_config):
        conn = connect(tmp_config)
        tables = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "documents" in tables
        assert "chunks" in tables
        assert "meta" in tables
        conn.close()

    def test_creates_virtual_tables(self, tmp_config):
        conn = connect(tmp_config)
        tables = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "vec_chunks" in tables
        assert "fts_chunks" in tables
        conn.close()

    def test_sets_schema_version(self, tmp_config):
        conn = connect(tmp_config)
        row = conn.execute(
            "SELECT value FROM meta WHERE key = 'schema_version'"
        ).fetchone()
        assert int(row[0]) == SCHEMA_VERSION
        conn.close()

    def test_idempotent_on_second_call(self, tmp_config):
        conn1 = connect(tmp_config)
        conn1.execute(
            "INSERT INTO documents (path, title, type, size_bytes, content_hash) "
            "VALUES ('test.md', 'Test', 'markdown', 100, 'abc')"
        )
        conn1.commit()
        conn1.close()

        conn2 = connect(tmp_config)
        count = conn2.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        assert count == 1
        conn2.close()

    def test_schema_upgrade_drops_tables(self, tmp_config):
        # Create DB with old schema version
        tmp_config.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(tmp_config.db_path))
        conn.execute(
            "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)"
        )
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_version', '1')"
        )
        conn.execute("CREATE TABLE IF NOT EXISTS documents (id INTEGER PRIMARY KEY)")
        conn.execute("INSERT INTO documents VALUES (1)")
        conn.commit()
        conn.close()

        # Reconnect — should upgrade
        conn2 = connect(tmp_config)
        count = conn2.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        assert count == 0  # table was recreated
        conn2.close()

    def test_creates_parent_dirs(self, tmp_path):
        cfg = Config()
        cfg.db_path = tmp_path / "sub" / "deep" / "kb.db"
        cfg.embed_dims = 1536
        conn = connect(cfg)
        assert cfg.db_path.exists()
        conn.close()

    def test_v3_to_v4_migration_adds_tags(self, tmp_config):
        """v3 -> v4 uses ALTER TABLE instead of dropping tables."""
        import sqlite_vec

        # Create a v3 DB with data
        tmp_config.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(tmp_config.db_path))
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.execute(
            "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)"
        )
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_version', '3')"
        )
        conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT UNIQUE NOT NULL,
                title TEXT,
                type TEXT,
                size_bytes INTEGER,
                content_hash TEXT,
                indexed_at TEXT DEFAULT (datetime('now')),
                chunk_count INTEGER DEFAULT 0
            )
        """)
        conn.execute(
            "INSERT INTO documents (path, title, type, size_bytes, content_hash, chunk_count) "
            "VALUES ('test.md', 'Test', 'markdown', 100, 'abc', 1)"
        )
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id INTEGER NOT NULL,
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
                embedding float[{tmp_config.embed_dims}],
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
        conn.close()

        # Reconnect — should migrate v3->v4 without dropping data
        conn2 = connect(tmp_config)
        count = conn2.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        assert count == 1  # data preserved
        # tags column exists
        row = conn2.execute(
            "SELECT tags FROM documents WHERE path = 'test.md'"
        ).fetchone()
        assert row["tags"] == ""
        # Schema version updated
        version = conn2.execute(
            "SELECT value FROM meta WHERE key = 'schema_version'"
        ).fetchone()
        assert int(version[0]) == SCHEMA_VERSION
        conn2.close()

    def test_row_factory_enabled(self, tmp_config):
        conn = connect(tmp_config)
        conn.execute(
            "INSERT INTO documents (path, title, type, size_bytes, content_hash) "
            "VALUES ('test.md', 'Test', 'markdown', 100, 'abc')"
        )
        conn.commit()
        row = conn.execute("SELECT * FROM documents").fetchone()
        assert row["path"] == "test.md"
        conn.close()

    def test_v4_to_v5_migration_rebuilds_fts(self, tmp_config):
        """v4 -> v5 drops fts_chunks and recreates with triggers."""
        import sqlite_vec

        tmp_config.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(tmp_config.db_path))
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.execute(
            "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)"
        )
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_version', '4')"
        )
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
        conn.execute(
            "INSERT INTO documents (path, title, type, size_bytes, content_hash, chunk_count) "
            "VALUES ('test.md', 'Test', 'markdown', 100, 'abc', 1)"
        )
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id INTEGER NOT NULL,
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
                embedding float[{tmp_config.embed_dims}],
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
        conn.close()

        # Reconnect — should migrate v4->v5
        conn2 = connect(tmp_config)
        # Data preserved
        count = conn2.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        assert count == 1
        # Schema version updated
        version = conn2.execute(
            "SELECT value FROM meta WHERE key = 'schema_version'"
        ).fetchone()
        assert int(version[0]) == SCHEMA_VERSION
        # Triggers exist
        triggers = {
            r[0]
            for r in conn2.execute(
                "SELECT name FROM sqlite_master WHERE type='trigger'"
            ).fetchall()
        }
        assert "fts_ai" in triggers
        assert "fts_ad" in triggers
        assert "fts_au" in triggers
        conn2.close()

    def test_v5_to_v6_migration_adds_porter_tokenizer(self, tmp_config):
        """v5 -> v6 drops fts_chunks and recreates with porter tokenizer."""
        import sqlite_vec

        tmp_config.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(tmp_config.db_path))
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.execute(
            "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)"
        )
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_version', '5')"
        )
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
        conn.execute(
            "INSERT INTO documents (path, title, type, size_bytes, content_hash, chunk_count) "
            "VALUES ('test.md', 'Test', 'markdown', 100, 'abc', 1)"
        )
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id INTEGER NOT NULL,
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
                embedding float[{tmp_config.embed_dims}],
                +chunk_text TEXT,
                +doc_path TEXT,
                +heading TEXT
            )
        """)
        # Old FTS without porter tokenizer
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS fts_chunks USING fts5(
                text,
                heading,
                content='chunks',
                content_rowid='id'
            )
        """)
        conn.commit()
        conn.close()

        # Reconnect — should migrate v5->v6
        conn2 = connect(tmp_config)
        # Data preserved
        count = conn2.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        assert count == 1
        # Schema version updated
        version = conn2.execute(
            "SELECT value FROM meta WHERE key = 'schema_version'"
        ).fetchone()
        assert int(version[0]) == SCHEMA_VERSION
        # FTS table recreated (with porter tokenizer)
        fts_sql = conn2.execute(
            "SELECT sql FROM sqlite_master WHERE name = 'fts_chunks'"
        ).fetchone()[0]
        assert "porter" in fts_sql
        # Triggers exist
        triggers = {
            r[0]
            for r in conn2.execute(
                "SELECT name FROM sqlite_master WHERE type='trigger'"
            ).fetchall()
        }
        assert "fts_ai" in triggers
        assert "fts_ad" in triggers
        assert "fts_au" in triggers
        conn2.close()

    def test_fts_triggers_sync(self, tmp_config):
        """FTS triggers keep fts_chunks in sync with chunks table."""
        conn = connect(tmp_config)
        # Insert a document
        conn.execute(
            "INSERT INTO documents (path, title, type, size_bytes, content_hash) "
            "VALUES ('notes/test.md', 'Test', 'markdown', 100, 'abc')"
        )
        doc_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

        # Insert a chunk with doc_path — trigger should sync to FTS
        conn.execute(
            "INSERT INTO chunks (doc_id, chunk_index, text, heading, char_count, doc_path) "
            "VALUES (?, 0, 'hello world', 'Intro', 11, 'notes/test.md')",
            (doc_id,),
        )
        chunk_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.commit()

        # FTS should find by text
        fts_rows = conn.execute(
            "SELECT rowid FROM fts_chunks WHERE fts_chunks MATCH '\"hello\"'"
        ).fetchall()
        assert len(fts_rows) == 1
        assert fts_rows[0][0] == chunk_id

        # FTS should find by doc_path
        fts_rows = conn.execute(
            "SELECT rowid FROM fts_chunks WHERE fts_chunks MATCH 'doc_path:\"notes\"'"
        ).fetchall()
        assert len(fts_rows) == 1
        assert fts_rows[0][0] == chunk_id

        # Delete the chunk — trigger should remove from FTS
        conn.execute("DELETE FROM chunks WHERE id = ?", (chunk_id,))
        conn.commit()

        fts_rows = conn.execute(
            "SELECT rowid FROM fts_chunks WHERE fts_chunks MATCH '\"hello\"'"
        ).fetchall()
        assert len(fts_rows) == 0
        conn.close()

    def test_wal_mode_enabled(self, tmp_config):
        """WAL journal mode is set for better concurrent read performance."""
        conn = connect(tmp_config)
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode == "wal"
        conn.close()

    def test_foreign_keys_enabled(self, tmp_config):
        """Foreign keys pragma is enabled."""
        conn = connect(tmp_config)
        fk = conn.execute("PRAGMA foreign_keys").fetchone()[0]
        assert fk == 1
        conn.close()

    def test_v6_to_v7_migration_adds_doc_path_and_weighted_fts(self, tmp_config):
        """v6 -> v7 adds doc_path column to chunks and rebuilds FTS with 3 columns + weights."""
        import sqlite_vec

        tmp_config.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(tmp_config.db_path))
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.execute(
            "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)"
        )
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_version', '6')"
        )
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
        conn.execute(
            "INSERT INTO documents (path, title, type, size_bytes, content_hash, chunk_count) "
            "VALUES ('notes/guide.md', 'Guide', 'markdown', 200, 'def', 1)"
        )
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                text TEXT NOT NULL,
                heading TEXT,
                heading_ancestry TEXT,
                char_count INTEGER,
                content_hash TEXT
            )
        """)
        conn.execute(
            "INSERT INTO chunks (doc_id, chunk_index, text, heading, char_count) "
            "VALUES (1, 0, 'some content', 'Intro', 12)"
        )
        conn.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
                chunk_id INTEGER PRIMARY KEY,
                embedding float[{tmp_config.embed_dims}],
                +chunk_text TEXT,
                +doc_path TEXT,
                +heading TEXT
            )
        """)
        # Old 2-column FTS (v6 schema)
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS fts_chunks USING fts5(
                text,
                heading,
                content='chunks',
                content_rowid='id',
                tokenize='porter unicode61'
            )
        """)
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS fts_ai AFTER INSERT ON chunks BEGIN
                INSERT INTO fts_chunks(rowid, text, heading)
                VALUES (new.id, new.text, new.heading);
            END
        """)
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS fts_ad AFTER DELETE ON chunks BEGIN
                INSERT INTO fts_chunks(fts_chunks, rowid, text, heading)
                VALUES ('delete', old.id, old.text, old.heading);
            END
        """)
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS fts_au AFTER UPDATE ON chunks BEGIN
                INSERT INTO fts_chunks(fts_chunks, rowid, text, heading)
                VALUES ('delete', old.id, old.text, old.heading);
                INSERT INTO fts_chunks(rowid, text, heading)
                VALUES (new.id, new.text, new.heading);
            END
        """)
        conn.commit()
        conn.close()

        # Reconnect — should migrate v6->v7
        conn2 = connect(tmp_config)

        # Data preserved
        count = conn2.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        assert count == 1

        # Schema version updated
        version = conn2.execute(
            "SELECT value FROM meta WHERE key = 'schema_version'"
        ).fetchone()
        assert int(version[0]) == SCHEMA_VERSION

        # doc_path column exists and is populated from documents.path
        row = conn2.execute("SELECT doc_path FROM chunks WHERE doc_id = 1").fetchone()
        assert row["doc_path"] == "notes/guide.md"

        # FTS table has 3 columns (doc_path, heading, text)
        fts_sql = conn2.execute(
            "SELECT sql FROM sqlite_master WHERE name = 'fts_chunks'"
        ).fetchone()[0]
        assert "doc_path" in fts_sql
        assert "heading" in fts_sql
        assert "text" in fts_sql
        assert "porter" in fts_sql

        # FTS rebuild populated the index — can find by text
        fts_rows = conn2.execute(
            "SELECT rowid FROM fts_chunks WHERE fts_chunks MATCH '\"content\"'"
        ).fetchall()
        assert len(fts_rows) == 1

        # FTS can find by doc_path
        fts_rows = conn2.execute(
            "SELECT rowid FROM fts_chunks WHERE fts_chunks MATCH 'doc_path:\"guide\"'"
        ).fetchall()
        assert len(fts_rows) == 1

        # Triggers exist with new schema
        triggers = {
            r[0]
            for r in conn2.execute(
                "SELECT name FROM sqlite_master WHERE type='trigger'"
            ).fetchall()
        }
        assert "fts_ai" in triggers
        assert "fts_ad" in triggers
        assert "fts_au" in triggers

        conn2.close()

    def test_fts_uses_porter_tokenizer(self, tmp_config):
        """FTS5 table uses porter unicode61 tokenizer for stemming."""
        conn = connect(tmp_config)
        fts_sql = conn.execute(
            "SELECT sql FROM sqlite_master WHERE name = 'fts_chunks'"
        ).fetchone()[0]
        assert "porter" in fts_sql
        assert "unicode61" in fts_sql
        conn.close()

    def test_triggers_created_on_fresh_db(self, tmp_config):
        """Fresh DB should have all three FTS triggers."""
        conn = connect(tmp_config)
        triggers = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='trigger'"
            ).fetchall()
        }
        assert triggers == {"fts_ai", "fts_ad", "fts_au"}
        conn.close()


class TestReset:
    def test_deletes_existing_db(self, tmp_path, capsys):
        db_path = tmp_path / "kb.db"
        db_path.write_text("fake db")
        reset(db_path)
        assert not db_path.exists()
        assert "Deleted" in capsys.readouterr().out

    def test_missing_db_prints_message(self, tmp_path, capsys):
        db_path = tmp_path / "kb.db"
        reset(db_path)
        assert "No database" in capsys.readouterr().out
