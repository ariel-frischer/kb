"""Tests for kb.cli — command dispatch and CLI commands."""

from pathlib import Path
from unittest.mock import patch

import pytest

from kb.cli import _best_snippet, cmd_add, cmd_init, cmd_remove, cmd_sources
from kb.config import (
    PROJECT_CONFIG_FILE,
    Config,
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


class TestBestSnippet:
    def test_empty_text(self):
        assert _best_snippet("", "hello") == ""

    def test_none_text(self):
        assert _best_snippet(None, "hello") == ""

    def test_short_text_returned_as_is(self):
        assert _best_snippet("short text", "short", width=100) == "short text"

    def test_empty_query(self):
        text = "a" * 600
        result = _best_snippet(text, "", width=100)
        assert result == text[:100]

    def test_no_match_returns_start(self):
        text = "a" * 600
        result = _best_snippet(text, "zzz", width=100)
        assert result == text[:100]

    def test_match_at_start_no_prefix(self):
        text = "hello world " + "x" * 600
        result = _best_snippet(text, "hello", width=100)
        assert result.startswith("hello")
        assert not result.startswith("...")

    def test_match_in_middle_has_prefix(self):
        text = "x" * 300 + " NEEDLE " + "y" * 300
        result = _best_snippet(text, "needle", width=100)
        assert "..." in result
        assert "NEEDLE" in result

    def test_match_centered(self):
        text = "a" * 300 + "MATCH" + "b" * 300
        result = _best_snippet(text, "match", width=100)
        assert "MATCH" in result
        # Match should be roughly centered — not at the edges
        idx = result.index("MATCH")
        assert idx > 10
        assert idx < 80

    def test_match_at_end(self):
        text = "x" * 500 + "FINDME"
        result = _best_snippet(text, "findme", width=100)
        assert "FINDME" in result
        assert result.endswith("FINDME")
        assert result.startswith("...")

    def test_match_at_end_no_suffix(self):
        text = "x" * 500 + "FINDME"
        result = _best_snippet(text, "findme", width=100)
        assert not result.endswith("...")

    def test_text_exactly_width(self):
        text = "a" * 500
        result = _best_snippet(text, "zzz", width=500)
        assert result == text

    def test_multiple_query_words_centers_between(self):
        text = "a" * 200 + "FIRST" + "b" * 100 + "SECOND" + "c" * 200
        result = _best_snippet(text, "first second", width=200)
        assert "FIRST" in result or "SECOND" in result

    def test_case_insensitive(self):
        text = "x" * 300 + "HeLLo WoRLd" + "y" * 300
        result = _best_snippet(text, "hello world", width=100)
        assert "HeLLo WoRLd" in result

    def test_special_chars_in_query(self):
        text = "x" * 300 + "some text here" + "y" * 300
        result = _best_snippet(text, '"some text"', width=100)
        assert "some text" in result

    def test_width_1_no_crash(self):
        text = "abc"
        result = _best_snippet(text, "b", width=1)
        assert len(result) >= 1

    def test_single_char_text(self):
        assert _best_snippet("a", "a") == "a"

    def test_query_with_no_alphanumeric(self):
        text = "x" * 600
        result = _best_snippet(text, "!@#$%", width=100)
        assert result == text[:100]


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
        with (
            patch("sys.argv", ["kb", "bogus"]),
            patch("kb.cli.find_config", return_value=Config()),
        ):
            with pytest.raises(SystemExit):
                from kb.cli import main

                main()
