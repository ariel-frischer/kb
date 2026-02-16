"""Tests for kb.cli — command dispatch and CLI commands."""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from kb.cli import cmd_add, cmd_init, cmd_remove, cmd_sources
from kb.config import (
    GLOBAL_CONFIG_TEMPLATE,
    PROJECT_CONFIG_FILE,
    PROJECT_CONFIG_TEMPLATE,
    Config,
    _load_toml,
)


class TestCmdInit:
    def test_init_project(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        cmd_init(project=True)
        cfg_path = tmp_path / PROJECT_CONFIG_FILE
        assert cfg_path.exists()
        content = cfg_path.read_text()
        assert "sources" in content

    def test_init_project_already_exists(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / PROJECT_CONFIG_FILE).write_text("existing")
        with pytest.raises(SystemExit):
            cmd_init(project=True)

    def test_init_global(self, tmp_path, monkeypatch):
        global_dir = tmp_path / "config"
        global_file = global_dir / "config.toml"
        data_dir = tmp_path / "data"
        monkeypatch.setattr("kb.cli.GLOBAL_CONFIG_DIR", global_dir)
        monkeypatch.setattr("kb.cli.GLOBAL_CONFIG_FILE", global_file)
        monkeypatch.setattr("kb.cli.GLOBAL_DATA_DIR", data_dir)
        cmd_init(project=False)
        assert global_file.exists()
        assert data_dir.exists()

    def test_init_global_already_exists(self, tmp_path, monkeypatch):
        global_file = tmp_path / "config.toml"
        global_file.write_text("existing")
        monkeypatch.setattr("kb.cli.GLOBAL_CONFIG_FILE", global_file)
        with pytest.raises(SystemExit):
            cmd_init(project=False)


class TestCmdAdd:
    def test_adds_directory(self, tmp_path):
        cfg = Config()
        cfg.scope = "project"
        cfg.config_dir = tmp_path
        cfg.config_path = tmp_path / ".kb.toml"

        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()

        cmd_add(cfg, [str(docs_dir)])
        assert "docs" in cfg.sources

    def test_adds_multiple_dirs(self, tmp_path):
        cfg = Config()
        cfg.scope = "project"
        cfg.config_dir = tmp_path
        cfg.config_path = tmp_path / ".kb.toml"

        for name in ["docs", "notes"]:
            (tmp_path / name).mkdir()

        cmd_add(cfg, [str(tmp_path / "docs"), str(tmp_path / "notes")])
        assert len(cfg.sources) == 2

    def test_rejects_nonexistent_dir(self, tmp_path):
        cfg = Config()
        cfg.scope = "project"
        cfg.config_dir = tmp_path
        cfg.config_path = tmp_path / ".kb.toml"

        with pytest.raises(SystemExit):
            cmd_add(cfg, [str(tmp_path / "nonexistent")])

    def test_skips_duplicate(self, tmp_path, capsys):
        cfg = Config(sources=["docs"])
        cfg.scope = "project"
        cfg.config_dir = tmp_path
        cfg.config_path = tmp_path / ".kb.toml"

        (tmp_path / "docs").mkdir()
        cmd_add(cfg, [str(tmp_path / "docs")])
        assert cfg.sources.count("docs") == 1
        assert "Already added" in capsys.readouterr().out

    def test_no_args_exits(self, tmp_path):
        cfg = Config()
        cfg.config_path = tmp_path / ".kb.toml"
        with pytest.raises(SystemExit):
            cmd_add(cfg, [])

    def test_global_uses_absolute_path(self, tmp_path):
        cfg = Config()
        cfg.scope = "global"
        cfg.config_dir = tmp_path
        cfg.config_path = tmp_path / "config.toml"

        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()

        cmd_add(cfg, [str(docs_dir)])
        assert cfg.sources[0] == str(docs_dir)


class TestCmdRemove:
    def test_removes_directory(self, tmp_path):
        cfg = Config(sources=["docs"])
        cfg.scope = "project"
        cfg.config_dir = tmp_path
        cfg.config_path = tmp_path / ".kb.toml"

        (tmp_path / "docs").mkdir()
        cmd_remove(cfg, [str(tmp_path / "docs")])
        assert cfg.sources == []

    def test_not_found_prints_message(self, tmp_path, capsys):
        cfg = Config(sources=["docs"])
        cfg.scope = "project"
        cfg.config_dir = tmp_path
        cfg.config_path = tmp_path / ".kb.toml"

        (tmp_path / "other").mkdir()
        cmd_remove(cfg, [str(tmp_path / "other")])
        assert "Not found" in capsys.readouterr().out

    def test_no_args_exits(self, tmp_path):
        cfg = Config()
        cfg.config_path = tmp_path / ".kb.toml"
        with pytest.raises(SystemExit):
            cmd_remove(cfg, [])


class TestCmdSources:
    def test_lists_sources(self, tmp_path, capsys):
        cfg = Config(sources=["docs", "notes"])
        cfg.scope = "project"
        cfg.config_dir = tmp_path

        (tmp_path / "docs").mkdir()
        cmd_sources(cfg)

        output = capsys.readouterr().out
        assert "docs" in output
        assert "notes" in output
        assert "(missing)" in output  # notes dir doesn't exist

    def test_no_sources(self, capsys):
        cfg = Config()
        cfg.scope = "project"
        cfg.config_dir = Path("/tmp")
        cmd_sources(cfg)
        assert "No sources" in capsys.readouterr().out


class TestMainDispatch:
    def test_no_args_prints_usage(self, capsys):
        with patch("sys.argv", ["kb"]):
            with pytest.raises(SystemExit):
                from kb.cli import main
                main()

    def test_help_flag(self, capsys):
        with patch("sys.argv", ["kb", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                from kb.cli import main
                main()
            assert exc_info.value.code == 0

    def test_unknown_command(self, capsys):
        with patch("sys.argv", ["kb", "bogus"]), \
             patch("kb.cli.find_config", return_value=Config()):
            with pytest.raises(SystemExit):
                from kb.cli import main
                main()
