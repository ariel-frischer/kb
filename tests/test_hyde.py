"""Tests for kb.hyde — HyDE (Hypothetical Document Embeddings)."""

from unittest.mock import MagicMock

from kb.config import Config
from kb.hyde import generate_hyde_passage


def _mock_client(content="A hypothetical passage about the topic."):
    client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock()]
    mock_resp.choices[0].message.content = content
    client.chat.completions.create.return_value = mock_resp
    return client


class TestGenerateHydePassage:
    def test_returns_passage_and_timing(self):
        client = _mock_client("This is a hypothetical answer.")
        cfg = Config(chat_model="gpt-4o-mini")

        passage, elapsed = generate_hyde_passage("what is kb?", client, cfg)

        assert passage == "This is a hypothetical answer."
        assert elapsed > 0
        client.chat.completions.create.assert_called_once()

    def test_uses_hyde_model_when_set(self):
        client = _mock_client()
        cfg = Config(chat_model="gpt-4o-mini", hyde_model="gpt-4o")

        generate_hyde_passage("test query", client, cfg)

        call_kwargs = client.chat.completions.create.call_args
        assert call_kwargs.kwargs["model"] == "gpt-4o"

    def test_falls_back_to_chat_model_when_hyde_model_empty(self):
        client = _mock_client()
        cfg = Config(chat_model="gpt-4o-mini", hyde_model="")

        generate_hyde_passage("test query", client, cfg)

        call_kwargs = client.chat.completions.create.call_args
        assert call_kwargs.kwargs["model"] == "gpt-4o-mini"

    def test_returns_none_on_api_error(self):
        client = MagicMock()
        client.chat.completions.create.side_effect = RuntimeError("API error")
        cfg = Config()

        passage, elapsed = generate_hyde_passage("test", client, cfg)

        assert passage is None
        assert elapsed > 0

    def test_returns_none_on_empty_response(self):
        client = _mock_client("")
        cfg = Config()

        passage, elapsed = generate_hyde_passage("test", client, cfg)

        assert passage is None
        assert elapsed > 0

    def test_strips_whitespace_from_response(self):
        client = _mock_client("  passage with spaces  \n")
        cfg = Config()

        passage, _ = generate_hyde_passage("test", client, cfg)

        assert passage == "passage with spaces"

    def test_returns_none_when_content_is_none(self):
        client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = None
        client.chat.completions.create.return_value = mock_resp
        cfg = Config()

        passage, elapsed = generate_hyde_passage("test", client, cfg)

        assert passage is None

    def test_passes_correct_parameters(self):
        client = _mock_client()
        cfg = Config(chat_model="gpt-4o-mini")

        generate_hyde_passage("what is python?", client, cfg)

        call_kwargs = client.chat.completions.create.call_args.kwargs
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["max_tokens"] == 300
        assert len(call_kwargs["messages"]) == 2
        assert call_kwargs["messages"][1]["content"] == "what is python?"
