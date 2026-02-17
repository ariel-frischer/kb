"""Tests for kb.rerank — LLM and cross-encoder reranking logic."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from kb.config import Config
from kb.rerank import (
    _cross_encoder_cache,
    cross_encoder_rerank,
    llm_rerank,
    rerank,
)


def _make_results(n):
    return [
        {
            "chunk_id": i,
            "text": f"Passage {i} content",
            "doc_path": f"doc{i}.md",
            "heading": f"Section {i}",
            "similarity": 0.9 - i * 0.1,
        }
        for i in range(n)
    ]


class TestLlmRerank:
    def test_returns_unchanged_if_fewer_than_top_k(self):
        cfg = Config(rerank_top_k=5)
        client = MagicMock()
        results = _make_results(3)
        out, info = llm_rerank(client, "query", results, cfg)
        assert out == results
        assert info == {}
        client.chat.completions.create.assert_not_called()

    def test_reranks_by_llm_output(self):
        cfg = Config(rerank_top_k=3)
        client = MagicMock()

        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "3, 1, 5, 2, 4, 6"
        mock_resp.usage = MagicMock(prompt_tokens=100, completion_tokens=20)
        client.chat.completions.create.return_value = mock_resp

        results = _make_results(6)
        out, info = llm_rerank(client, "query", results, cfg)

        assert len(out) == 3
        # LLM said 3,1,5 so indices 2,0,4
        assert out[0]["chunk_id"] == 2
        assert out[1]["chunk_id"] == 0
        assert out[2]["chunk_id"] == 4
        assert info["input_count"] == 6
        assert info["output_count"] == 3

    def test_handles_partial_ranking(self):
        """LLM doesn't mention all passages — missing ones appended."""
        cfg = Config(rerank_top_k=4)
        client = MagicMock()

        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "2, 4"  # only mentions 2 of 5
        mock_resp.usage = MagicMock(prompt_tokens=50, completion_tokens=10)
        client.chat.completions.create.return_value = mock_resp

        results = _make_results(5)
        out, info = llm_rerank(client, "query", results, cfg)

        assert len(out) == 4
        # First two are the LLM picks (indices 1, 3)
        assert out[0]["chunk_id"] == 1
        assert out[1]["chunk_id"] == 3

    def test_ignores_out_of_range_indices(self):
        cfg = Config(rerank_top_k=3)
        client = MagicMock()

        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "99, 1, 2, 3"  # 99 is out of range
        mock_resp.usage = MagicMock(prompt_tokens=50, completion_tokens=10)
        client.chat.completions.create.return_value = mock_resp

        results = _make_results(4)
        out, info = llm_rerank(client, "query", results, cfg)

        assert len(out) == 3
        assert out[0]["chunk_id"] == 0  # index 0 from "1"

    def test_uses_correct_model(self):
        cfg = Config(rerank_top_k=2, chat_model="gpt-4o")
        client = MagicMock()

        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "1, 2, 3"
        mock_resp.usage = MagicMock(prompt_tokens=50, completion_tokens=10)
        client.chat.completions.create.return_value = mock_resp

        llm_rerank(client, "test question", _make_results(4), cfg)

        call_kwargs = client.chat.completions.create.call_args
        assert call_kwargs.kwargs["model"] == "gpt-4o"
        assert call_kwargs.kwargs["temperature"] == 0


class TestCrossEncoderRerank:
    def setup_method(self):
        _cross_encoder_cache.clear()

    def test_returns_unchanged_if_fewer_than_top_k(self):
        cfg = Config(rerank_top_k=5, rerank_method="cross-encoder")
        results = _make_results(3)
        out, info = cross_encoder_rerank("query", results, cfg)
        assert out == results
        assert info == {}

    @patch("kb.rerank._get_cross_encoder")
    def test_reranks_by_score(self, mock_get_ce):
        cfg = Config(rerank_top_k=3, rerank_method="cross-encoder")
        mock_model = MagicMock()
        # Scores: passage 0=0.1, 1=0.9, 2=0.5, 3=0.3, 4=0.7
        mock_model.predict.return_value = np.array([0.1, 0.9, 0.5, 0.3, 0.7])
        mock_get_ce.return_value = mock_model

        results = _make_results(5)
        out, info = cross_encoder_rerank("query", results, cfg)

        assert len(out) == 3
        # Sorted by score desc: 1 (0.9), 4 (0.7), 2 (0.5)
        assert out[0]["chunk_id"] == 1
        assert out[1]["chunk_id"] == 4
        assert out[2]["chunk_id"] == 2
        assert info["input_count"] == 5
        assert info["output_count"] == 3
        assert "rerank_ms" in info
        assert info["model"] == cfg.cross_encoder_model

    @patch("kb.rerank._get_cross_encoder")
    def test_includes_heading_in_passage(self, mock_get_ce):
        cfg = Config(rerank_top_k=1, rerank_method="cross-encoder")
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.8, 0.2])
        mock_get_ce.return_value = mock_model

        results = _make_results(2)
        cross_encoder_rerank("test query", results, cfg)

        pairs = mock_model.predict.call_args[0][0]
        # Each pair should be (question, passage_with_heading)
        assert pairs[0][0] == "test query"
        assert "Section 0" in pairs[0][1]
        assert "Passage 0 content" in pairs[0][1]

    @patch("kb.rerank._get_cross_encoder")
    def test_handles_missing_heading(self, mock_get_ce):
        cfg = Config(rerank_top_k=1, rerank_method="cross-encoder")
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.5, 0.3])
        mock_get_ce.return_value = mock_model

        results = [
            {"chunk_id": 0, "text": "content", "doc_path": "a.md", "heading": ""},
            {"chunk_id": 1, "text": "other", "doc_path": "b.md", "heading": None},
        ]
        cross_encoder_rerank("query", results, cfg)

        pairs = mock_model.predict.call_args[0][0]
        assert pairs[0][1] == "content"
        assert pairs[1][1] == "other"

    @patch("kb.rerank._get_cross_encoder")
    def test_raises_import_error_when_missing(self, mock_get_ce):
        mock_get_ce.side_effect = ImportError("sentence-transformers is required")
        cfg = Config(rerank_top_k=2, rerank_method="cross-encoder")
        with pytest.raises(ImportError, match="sentence-transformers"):
            cross_encoder_rerank("query", _make_results(3), cfg)


class TestRerankDispatcher:
    def setup_method(self):
        _cross_encoder_cache.clear()

    def test_dispatches_to_llm_by_default(self):
        cfg = Config(rerank_top_k=5)
        client = MagicMock()
        results = _make_results(3)
        out, info = rerank(client, "query", results, cfg)
        assert out == results  # fewer than top_k, returns unchanged

    @patch("kb.rerank.cross_encoder_rerank")
    def test_dispatches_to_cross_encoder(self, mock_ce_rerank):
        cfg = Config(rerank_method="cross-encoder")
        mock_ce_rerank.return_value = ([], {})
        rerank(None, "query", _make_results(3), cfg)
        mock_ce_rerank.assert_called_once()

    @patch("kb.rerank.llm_rerank")
    def test_dispatches_to_llm_explicitly(self, mock_llm_rerank):
        cfg = Config(rerank_method="llm")
        client = MagicMock()
        mock_llm_rerank.return_value = ([], {})
        rerank(client, "query", _make_results(3), cfg)
        mock_llm_rerank.assert_called_once()
