"""Tests for kb feedback — API, CLI, and YAML helpers."""

from unittest.mock import patch

import pytest

from kb.api import (
    KBError,
    _feedback_entry_to_yaml,
    _parse_feedback_yaml,
    _yaml_escape,
    feedback_core,
    list_feedback_core,
)
from kb.cli import cmd_feedback


# ---------------------------------------------------------------------------
# YAML helpers
# ---------------------------------------------------------------------------


class TestYamlEscape:
    def test_plain_string(self):
        assert _yaml_escape("hello") == "hello"

    def test_empty_string(self):
        assert _yaml_escape("") == '""'

    def test_colon(self):
        assert _yaml_escape("key: value") == '"key: value"'

    def test_newline(self):
        assert _yaml_escape("line1\nline2") == '"line1\\nline2"'

    def test_quotes(self):
        result = _yaml_escape('say "hello"')
        assert result == '"say \\"hello\\""'

    def test_hash(self):
        assert _yaml_escape("# comment") == '"# comment"'


class TestYamlRoundtrip:
    def test_simple_entry(self):
        entry = {
            "timestamp": "2026-01-01T00:00:00Z",
            "kb_version": "1.0.0",
            "message": "test message",
            "severity": "note",
        }
        yaml_str = _feedback_entry_to_yaml(entry)
        parsed = _parse_feedback_yaml(yaml_str)
        assert len(parsed) == 1
        assert parsed[0]["message"] == "test message"
        assert parsed[0]["severity"] == "note"
        assert parsed[0]["kb_version"] == "1.0.0"

    def test_special_chars_roundtrip(self):
        entry = {
            "timestamp": "2026-01-01T00:00:00Z",
            "kb_version": "1.0.0",
            "message": 'query "test" with: colons & newline\nand more',
            "severity": "bug",
            "tool": "kb_search",
        }
        yaml_str = _feedback_entry_to_yaml(entry)
        parsed = _parse_feedback_yaml(yaml_str)
        assert len(parsed) == 1
        assert parsed[0]["message"] == 'query "test" with: colons & newline\nand more'
        assert parsed[0]["tool"] == "kb_search"

    def test_multiple_entries(self):
        e1 = {
            "timestamp": "2026-01-01T00:00:00Z",
            "kb_version": "1.0.0",
            "message": "first",
            "severity": "note",
        }
        e2 = {
            "timestamp": "2026-01-02T00:00:00Z",
            "kb_version": "1.0.0",
            "message": "second",
            "severity": "bug",
            "tool": "kb_ask",
        }
        yaml_str = _feedback_entry_to_yaml(e1) + _feedback_entry_to_yaml(e2)
        parsed = _parse_feedback_yaml(yaml_str)
        assert len(parsed) == 2
        assert parsed[0]["message"] == "first"
        assert parsed[1]["message"] == "second"
        assert parsed[1]["tool"] == "kb_ask"


# ---------------------------------------------------------------------------
# API: feedback_core / list_feedback_core
# ---------------------------------------------------------------------------


class TestFeedbackCore:
    def test_submit(self, tmp_path):
        fb_path = tmp_path / "feedback.yml"
        with patch("kb.api.FEEDBACK_PATH", fb_path):
            entry = feedback_core("test bug report")
        assert entry["message"] == "test bug report"
        assert entry["severity"] == "note"
        assert entry["timestamp"]
        assert entry["kb_version"]
        assert fb_path.exists()

    def test_all_fields(self, tmp_path):
        fb_path = tmp_path / "feedback.yml"
        with patch("kb.api.FEEDBACK_PATH", fb_path):
            entry = feedback_core(
                "search failed",
                tool="kb_search",
                severity="bug",
                context="query was 'test'",
                agent_id="claude-code-1",
                error_trace="Traceback: ...",
            )
        assert entry["tool"] == "kb_search"
        assert entry["severity"] == "bug"
        assert entry["context"] == "query was 'test'"
        assert entry["agent_id"] == "claude-code-1"
        assert entry["error_trace"] == "Traceback: ..."

    def test_invalid_severity(self, tmp_path):
        fb_path = tmp_path / "feedback.yml"
        with patch("kb.api.FEEDBACK_PATH", fb_path):
            with pytest.raises(KBError, match="Invalid severity"):
                feedback_core("msg", severity="critical")

    def test_empty_message(self, tmp_path):
        fb_path = tmp_path / "feedback.yml"
        with patch("kb.api.FEEDBACK_PATH", fb_path):
            with pytest.raises(KBError, match="cannot be empty"):
                feedback_core("")

    def test_whitespace_only_message(self, tmp_path):
        fb_path = tmp_path / "feedback.yml"
        with patch("kb.api.FEEDBACK_PATH", fb_path):
            with pytest.raises(KBError, match="cannot be empty"):
                feedback_core("   ")

    def test_multiple_appends(self, tmp_path):
        fb_path = tmp_path / "feedback.yml"
        with patch("kb.api.FEEDBACK_PATH", fb_path):
            feedback_core("first")
            feedback_core("second")
            result = list_feedback_core()
        assert result["count"] == 2
        assert result["entries"][0]["message"] == "first"
        assert result["entries"][1]["message"] == "second"


class TestListFeedbackCore:
    def test_no_file(self, tmp_path):
        fb_path = tmp_path / "feedback.yml"
        with patch("kb.api.FEEDBACK_PATH", fb_path):
            result = list_feedback_core()
        assert result["count"] == 0
        assert result["entries"] == []

    def test_empty_file(self, tmp_path):
        fb_path = tmp_path / "feedback.yml"
        fb_path.write_text("")
        with patch("kb.api.FEEDBACK_PATH", fb_path):
            result = list_feedback_core()
        assert result["count"] == 0

    def test_list_entries(self, tmp_path):
        fb_path = tmp_path / "feedback.yml"
        with patch("kb.api.FEEDBACK_PATH", fb_path):
            feedback_core("alpha", severity="bug")
            feedback_core("beta", severity="suggestion", tool="kb_ask")
            result = list_feedback_core()
        assert result["count"] == 2
        assert result["entries"][0]["severity"] == "bug"
        assert result["entries"][1]["tool"] == "kb_ask"


# ---------------------------------------------------------------------------
# CLI: cmd_feedback
# ---------------------------------------------------------------------------


class TestCmdFeedback:
    def test_submit_output(self, tmp_path, capsys):
        fb_path = tmp_path / "feedback.yml"
        with patch("kb.api.FEEDBACK_PATH", fb_path):
            cmd_feedback(["test message"])
        out = capsys.readouterr().out
        assert "Feedback recorded" in out
        assert "test message" in out

    def test_submit_with_flags(self, tmp_path, capsys):
        fb_path = tmp_path / "feedback.yml"
        with patch("kb.api.FEEDBACK_PATH", fb_path):
            cmd_feedback(["bug report", "--severity", "bug", "--tool", "kb_search"])
        out = capsys.readouterr().out
        assert "[bug]" in out

    def test_list_empty(self, tmp_path, capsys):
        fb_path = tmp_path / "feedback.yml"
        with patch("kb.api.FEEDBACK_PATH", fb_path):
            cmd_feedback(["--list"])
        out = capsys.readouterr().out
        assert "No feedback entries" in out

    def test_list_entries(self, tmp_path, capsys):
        fb_path = tmp_path / "feedback.yml"
        with patch("kb.api.FEEDBACK_PATH", fb_path):
            feedback_core("entry one", severity="bug")
            feedback_core("entry two", tool="kb_ask")
            cmd_feedback(["--list"])
        out = capsys.readouterr().out
        assert "2 feedback entries" in out
        assert "entry one" in out
        assert "entry two" in out

    def test_no_message_exits(self, tmp_path):
        fb_path = tmp_path / "feedback.yml"
        with patch("kb.api.FEEDBACK_PATH", fb_path):
            with pytest.raises(SystemExit):
                cmd_feedback([])

    def test_unknown_flag_exits(self, tmp_path):
        fb_path = tmp_path / "feedback.yml"
        with patch("kb.api.FEEDBACK_PATH", fb_path):
            with pytest.raises(SystemExit):
                cmd_feedback(["msg", "--unknown"])

    def test_invalid_severity_exits(self, tmp_path, capsys):
        fb_path = tmp_path / "feedback.yml"
        with patch("kb.api.FEEDBACK_PATH", fb_path):
            with pytest.raises(SystemExit):
                cmd_feedback(["msg", "--severity", "critical"])
