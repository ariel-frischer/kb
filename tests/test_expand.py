"""Tests for kb.expand — query expansion (LLM + local)."""

import json
from unittest.mock import MagicMock, patch

import pytest

from kb.config import Config
from kb.expand import expand_query, llm_expand, local_expand


@pytest.fixture
def cfg():
    return Config(expand_method="llm", expand_model="google/flan-t5-small")


@pytest.fixture
def mock_client():
    return MagicMock()


class TestLlmExpand:
    def test_parses_json(self, mock_client, cfg):
        resp = MagicMock()
        resp.choices = [
            MagicMock(
                message=MagicMock(
                    content=json.dumps(
                        {
                            "lex": ["async coroutine", "await concurrency"],
                            "vec": ["how to use async in python"],
                        }
                    )
                )
            )
        ]
        mock_client.chat.completions.create.return_value = resp

        results = llm_expand(mock_client, "python async patterns", cfg)
        assert len(results) == 3
        lex = [r for r in results if r["type"] == "lex"]
        vec = [r for r in results if r["type"] == "vec"]
        assert len(lex) == 2
        assert len(vec) == 1
        assert lex[0]["text"] == "async coroutine"
        assert vec[0]["text"] == "how to use async in python"

    def test_filters_duplicates(self, mock_client, cfg):
        resp = MagicMock()
        resp.choices = [
            MagicMock(
                message=MagicMock(
                    content=json.dumps(
                        {
                            "lex": ["python async patterns", "coroutine"],
                            "vec": ["Python Async Patterns"],
                        }
                    )
                )
            )
        ]
        mock_client.chat.completions.create.return_value = resp

        results = llm_expand(mock_client, "python async patterns", cfg)
        texts = [r["text"] for r in results]
        assert "python async patterns" not in [t.lower() for t in texts]
        assert "coroutine" in texts

    def test_graceful_fallback(self, mock_client, cfg):
        mock_client.chat.completions.create.side_effect = Exception("API down")
        results = llm_expand(mock_client, "test query", cfg)
        assert results == []

    def test_malformed_json_fallback(self, mock_client, cfg):
        resp = MagicMock()
        resp.choices = [MagicMock(message=MagicMock(content="not json"))]
        mock_client.chat.completions.create.return_value = resp
        results = llm_expand(mock_client, "test query", cfg)
        assert results == []


class TestLocalExpand:
    def test_parses_json_from_model_output(self, cfg):
        cfg.expand_method = "local"

        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        mock_tokenizer.decode.return_value = (
            'Here you go: {"lex": ["coroutine", "Python Async"], '
            '"vec": ["how does async work in python"]}'
        )

        with patch(
            "kb.expand.load_causal_lm",
            return_value=(mock_tokenizer, mock_model, "cpu"),
        ):
            results = local_expand("python async", cfg)

        assert results == [
            {"type": "lex", "text": "coroutine"},
            {"type": "vec", "text": "how does async work in python"},
        ]

    def test_non_json_output_returns_empty(self, cfg):
        mock_tokenizer = MagicMock()
        mock_tokenizer.decode.return_value = "async coroutine, await pattern"

        with patch(
            "kb.expand.load_causal_lm",
            return_value=(mock_tokenizer, MagicMock(), "cpu"),
        ):
            assert local_expand("python async", cfg) == []

    def test_import_error(self, cfg):
        cfg.expand_method = "local"

        with patch(
            "kb.expand.load_causal_lm",
            side_effect=ImportError("transformers required"),
        ):
            with pytest.raises(ImportError, match="transformers"):
                local_expand("test query", cfg)


class TestExpandDispatch:
    def test_routes_to_llm(self, mock_client, cfg):
        cfg.expand_method = "llm"
        with patch(
            "kb.expand.llm_expand", return_value=[{"type": "lex", "text": "foo"}]
        ) as m:
            results, ms = expand_query(mock_client, "test", cfg)
            m.assert_called_once_with(mock_client, "test", cfg)
            assert results == [{"type": "lex", "text": "foo"}]
            assert ms >= 0

    def test_routes_to_local(self, mock_client, cfg):
        cfg.expand_method = "local"
        with patch(
            "kb.expand.local_expand", return_value=[{"type": "vec", "text": "bar"}]
        ) as m:
            results, ms = expand_query(mock_client, "test", cfg)
            m.assert_called_once_with("test", cfg)
            assert results == [{"type": "vec", "text": "bar"}]
