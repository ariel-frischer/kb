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
    def test_returns_typed_results(self, cfg):
        cfg.expand_method = "local"

        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        mock_tokenizer.return_value = {"input_ids": MagicMock()}
        mock_model.generate.return_value = [MagicMock()]
        mock_tokenizer.decode.side_effect = [
            "async coroutine, await pattern",
            "how does async work in python",
        ]

        with patch(
            "kb.expand._get_t5_model", return_value=(mock_tokenizer, mock_model)
        ):
            results = local_expand("python async", cfg)

        lex = [r for r in results if r["type"] == "lex"]
        vec = [r for r in results if r["type"] == "vec"]
        assert len(lex) >= 1
        assert len(vec) == 1
        assert all(r["text"] for r in results)

    def test_import_error(self, cfg):
        cfg.expand_method = "local"

        with patch("kb.expand._expand_model_cache", {}):
            with patch(
                "kb.expand._get_t5_model",
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
