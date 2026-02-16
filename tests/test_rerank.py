"""Tests for kb.rerank — LLM reranking logic."""

from unittest.mock import MagicMock

import pytest

from kb.config import Config
from kb.rerank import llm_rerank


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
        out = llm_rerank(client, "query", results, cfg)
        assert out == results
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
        out = llm_rerank(client, "query", results, cfg)

        assert len(out) == 3
        # LLM said 3,1,5 so indices 2,0,4
        assert out[0]["chunk_id"] == 2
        assert out[1]["chunk_id"] == 0
        assert out[2]["chunk_id"] == 4

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
        out = llm_rerank(client, "query", results, cfg)

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
        out = llm_rerank(client, "query", results, cfg)

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
