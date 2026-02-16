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
