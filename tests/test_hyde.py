"""Tests for kb.hyde — HyDE (Hypothetical Document Embeddings)."""

from unittest.mock import MagicMock, patch

from kb.config import Config
from kb.hyde import generate_hyde_passage


def _mock_client(content="A hypothetical passage about the topic."):
    client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock()]
    mock_resp.choices[0].message.content = content
    client.chat.completions.create.return_value = mock_resp
    return client


class TestLlmHydePassage:
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


class TestMethodDispatch:
    def test_llm_method_calls_openai(self):
        client = _mock_client("LLM passage")
        cfg = Config(hyde_method="llm")

        passage, elapsed = generate_hyde_passage("test", client, cfg)

        assert passage == "LLM passage"
        client.chat.completions.create.assert_called_once()

    @patch("kb.hyde._get_local_model")
    def test_local_method_calls_local_model(self, mock_get_model):
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        mock_get_model.return_value = (mock_tokenizer, mock_model, "cpu")

        mock_tokenizer.apply_chat_template.return_value = "formatted prompt"
        mock_inputs = MagicMock()
        mock_inputs.__getitem__ = MagicMock(
            return_value=MagicMock(shape=(1, 10))
        )
        mock_inputs.to.return_value = mock_inputs
        mock_tokenizer.return_value = mock_inputs

        import torch

        mock_model.generate.return_value = torch.tensor([[0] * 15])
        mock_tokenizer.decode.return_value = "Local passage about the topic."

        cfg = Config(hyde_method="local", hyde_local_model="test/model")

        passage, elapsed = generate_hyde_passage("test query", None, cfg)

        assert passage == "Local passage about the topic."
        assert elapsed > 0
        mock_get_model.assert_called_once_with("test/model")

    @patch("kb.hyde._get_local_model")
    def test_local_method_returns_none_on_empty(self, mock_get_model):
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        mock_get_model.return_value = (mock_tokenizer, mock_model, "cpu")

        mock_tokenizer.apply_chat_template.return_value = "formatted"
        mock_inputs = MagicMock()
        mock_inputs.__getitem__ = MagicMock(
            return_value=MagicMock(shape=(1, 5))
        )
        mock_inputs.to.return_value = mock_inputs
        mock_tokenizer.return_value = mock_inputs

        import torch

        mock_model.generate.return_value = torch.tensor([[0] * 8])
        mock_tokenizer.decode.return_value = "  "

        cfg = Config(hyde_method="local")

        passage, elapsed = generate_hyde_passage("test", None, cfg)

        assert passage is None
        assert elapsed > 0

    @patch("kb.hyde._get_local_model")
    def test_local_method_returns_none_on_generation_error(self, mock_get_model):
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        mock_get_model.return_value = (mock_tokenizer, mock_model, "cpu")

        mock_tokenizer.apply_chat_template.side_effect = RuntimeError("template error")

        cfg = Config(hyde_method="local")

        passage, elapsed = generate_hyde_passage("test", None, cfg)

        assert passage is None
        assert elapsed > 0

    @patch("kb.hyde._get_local_model")
    def test_local_method_returns_none_on_model_load_error(self, mock_get_model):
        mock_get_model.side_effect = RuntimeError("model load failed")

        cfg = Config(hyde_method="local")

        passage, elapsed = generate_hyde_passage("test", None, cfg)

        assert passage is None
        assert elapsed > 0

    def test_local_method_does_not_call_openai(self):
        client = _mock_client()
        cfg = Config(hyde_method="local")

        with patch("kb.hyde._get_local_model") as mock_get_model:
            mock_tokenizer = MagicMock()
            mock_model = MagicMock()
            mock_get_model.return_value = (mock_tokenizer, mock_model, "cpu")
            mock_tokenizer.apply_chat_template.return_value = "prompt"
            mock_inputs = MagicMock()
            mock_inputs.__getitem__ = MagicMock(
                return_value=MagicMock(shape=(1, 5))
            )
            mock_inputs.to.return_value = mock_inputs
            mock_tokenizer.return_value = mock_inputs

            import torch

            mock_model.generate.return_value = torch.tensor([[0] * 8])
            mock_tokenizer.decode.return_value = "local result"

            generate_hyde_passage("test", client, cfg)

        client.chat.completions.create.assert_not_called()
