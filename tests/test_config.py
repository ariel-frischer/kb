"""Tests for kb.config — Config dataclass, loading, saving, secrets."""

import os
from pathlib import Path

import pytest

from kb.config import Config, _load_toml, _to_toml, find_config, load_secrets, save_config


class TestConfigDataclass:
    def test_defaults(self):
        cfg = Config()
        assert cfg.embed_model == "text-embedding-3-small"
        assert cfg.embed_dims == 1536
        assert cfg.chat_model == "gpt-4o-mini"
        assert cfg.max_chunk_chars == 2000
        assert cfg.min_chunk_chars == 50
        assert cfg.min_similarity == 0.25
        assert cfg.rrf_k == 60.0
        assert cfg.rerank_fetch_k == 20
        assert cfg.rerank_top_k == 5
        assert cfg.scope == "project"
        assert cfg.sources == []

    def test_source_paths_project(self, tmp_path):
        cfg = Config(sources=["docs/", "notes/"])
        cfg.scope = "project"
        cfg.config_dir = tmp_path
        paths = cfg.source_paths
        assert paths == [tmp_path / "docs/", tmp_path / "notes/"]

    def test_source_paths_project_no_config_dir(self):
        cfg = Config(sources=["docs/"])
        cfg.scope = "project"
        cfg.config_dir = None
        assert cfg.source_paths == []

    def test_source_paths_global(self, tmp_path):
        cfg = Config(sources=[str(tmp_path / "notes"), "/tmp/docs"])
        cfg.scope = "global"
        paths = cfg.source_paths
        assert paths[0] == tmp_path / "notes"
        assert paths[1] == Path("/tmp/docs")

    def test_source_paths_global_tilde(self):
        cfg = Config(sources=["~/notes"])
        cfg.scope = "global"
        paths = cfg.source_paths
        assert paths[0] == Path("~/notes").expanduser()


class TestDocPathForDb:
    def test_project_mode_relative(self, tmp_path):
        cfg = Config()
        cfg.scope = "project"
        cfg.config_dir = tmp_path
        file_path = tmp_path / "docs" / "guide.md"
        source_dir = tmp_path / "docs"
        assert cfg.doc_path_for_db(file_path, source_dir) == "docs/guide.md"

    def test_project_mode_nested(self, tmp_path):
        cfg = Config()
        cfg.scope = "project"
        cfg.config_dir = tmp_path
        file_path = tmp_path / "docs" / "sub" / "deep.md"
        source_dir = tmp_path / "docs"
        assert cfg.doc_path_for_db(file_path, source_dir) == "docs/sub/deep.md"

    def test_global_mode(self, tmp_path):
        cfg = Config()
        cfg.scope = "global"
        cfg.config_dir = None
        source_dir = tmp_path / "notes"
        file_path = source_dir / "todo.md"
        assert cfg.doc_path_for_db(file_path, source_dir) == "notes/todo.md"

    def test_fallback_when_not_relative(self, tmp_path):
        cfg = Config()
        cfg.scope = "project"
        cfg.config_dir = tmp_path / "project"
        file_path = Path("/somewhere/else/file.md")
        source_dir = Path("/somewhere/else")
        assert cfg.doc_path_for_db(file_path, source_dir) == "else/file.md"


class TestLoadSecrets:
    def test_loads_keys_into_env(self, tmp_path, monkeypatch):
        secrets_file = tmp_path / "secrets.toml"
        secrets_file.write_text('openai_api_key = "sk-test-123"\n')
        monkeypatch.setattr("kb.config.SECRETS_PATH", secrets_file)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        load_secrets()
        assert os.environ["OPENAI_API_KEY"] == "sk-test-123"

    def test_existing_env_not_overwritten(self, tmp_path, monkeypatch):
        secrets_file = tmp_path / "secrets.toml"
        secrets_file.write_text('openai_api_key = "sk-from-file"\n')
        monkeypatch.setattr("kb.config.SECRETS_PATH", secrets_file)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")

        load_secrets()
        assert os.environ["OPENAI_API_KEY"] == "sk-from-env"

    def test_missing_file_is_noop(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kb.config.SECRETS_PATH", tmp_path / "nonexistent.toml")
        load_secrets()  # should not raise


class TestLoadToml:
    def test_loads_project_config(self, tmp_path):
        cfg_path = tmp_path / ".kb.toml"
        cfg_path.write_text(
            'db = "my.db"\n'
            'sources = ["docs/"]\n'
            'embed_model = "custom-model"\n'
        )
        cfg = _load_toml(cfg_path, "project")
        assert cfg.db == "my.db"
        assert cfg.sources == ["docs/"]
        assert cfg.embed_model == "custom-model"
        assert cfg.scope == "project"
        assert cfg.config_dir == tmp_path
        assert cfg.db_path == tmp_path / "my.db"

    def test_loads_global_config(self, tmp_path, monkeypatch):
        from kb.config import GLOBAL_DB_PATH

        cfg_path = tmp_path / "config.toml"
        cfg_path.write_text('sources = ["/home/user/notes"]\n')
        cfg = _load_toml(cfg_path, "global")
        assert cfg.scope == "global"
        assert cfg.db_path == GLOBAL_DB_PATH

    def test_ignores_unknown_keys(self, tmp_path):
        cfg_path = tmp_path / ".kb.toml"
        cfg_path.write_text('sources = []\nunknown_field = "ignored"\n')
        cfg = _load_toml(cfg_path, "project")
        assert cfg.sources == []
        assert not hasattr(cfg, "unknown_field")


class TestFindConfig:
    def test_finds_project_config(self, tmp_path, monkeypatch):
        cfg_path = tmp_path / ".kb.toml"
        cfg_path.write_text('sources = ["docs/"]\n')
        monkeypatch.chdir(tmp_path)
        cfg = find_config()
        assert cfg.scope == "project"
        assert cfg.config_path == cfg_path

    def test_walks_up_to_find_config(self, tmp_path, monkeypatch):
        cfg_path = tmp_path / ".kb.toml"
        cfg_path.write_text('sources = []\n')
        subdir = tmp_path / "sub" / "deep"
        subdir.mkdir(parents=True)
        monkeypatch.chdir(subdir)
        cfg = find_config()
        assert cfg.scope == "project"
        assert cfg.config_path == cfg_path

    def test_falls_back_to_global(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        global_cfg = tmp_path / "global_config.toml"
        global_cfg.write_text('sources = ["/notes"]\n')
        monkeypatch.setattr("kb.config.GLOBAL_CONFIG_FILE", global_cfg)
        cfg = find_config()
        assert cfg.scope == "global"

    def test_no_config_returns_defaults(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("kb.config.GLOBAL_CONFIG_FILE", tmp_path / "nope.toml")
        cfg = find_config()
        assert cfg.config_path is None
        assert cfg.scope == "project"


class TestToToml:
    def test_string_value(self):
        assert _to_toml({"key": "val"}) == 'key = "val"\n'

    def test_int_value(self):
        assert _to_toml({"num": 42}) == "num = 42\n"

    def test_float_value(self):
        assert _to_toml({"f": 3.14}) == "f = 3.14\n"

    def test_bool_value(self):
        assert _to_toml({"flag": True}) == "flag = true\n"
        assert _to_toml({"flag": False}) == "flag = false\n"

    def test_list_value(self):
        assert _to_toml({"items": ["a", "b"]}) == 'items = ["a", "b"]\n'

    def test_empty_list(self):
        assert _to_toml({"items": []}) == "items = []\n"

    def test_multiple_keys(self):
        result = _to_toml({"a": 1, "b": "two"})
        assert "a = 1" in result
        assert 'b = "two"' in result


class TestSaveConfig:
    def test_saves_non_default_values(self, tmp_path):
        cfg = Config(sources=["docs/"], embed_dims=768)
        cfg.config_path = tmp_path / "config.toml"
        cfg.config_dir = tmp_path
        save_config(cfg)

        content = cfg.config_path.read_text()
        assert "768" in content
        assert '"docs/"' in content
        # Default values should NOT be in the file
        assert "text-embedding-3-small" not in content

    def test_noop_without_config_path(self):
        cfg = Config()
        cfg.config_path = None
        save_config(cfg)  # should not raise

    def test_roundtrip(self, tmp_path):
        cfg = Config(sources=["notes/", "docs/"], max_chunk_chars=3000)
        cfg.config_path = tmp_path / ".kb.toml"
        cfg.config_dir = tmp_path
        save_config(cfg)

        loaded = _load_toml(cfg.config_path, "project")
        assert loaded.sources == ["notes/", "docs/"]
        assert loaded.max_chunk_chars == 3000
