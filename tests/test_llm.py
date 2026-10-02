"""Tests for the LLM provider helper (OpenAI API key vs ChatGPT subscription)."""

import base64
import json
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from kb import llm
from kb.api import KBError, _chat_cost_item
from kb.config import Config
from kb.cost import cost_summary
from kb.hyde import llm_hyde_passage_with_usage


def _jwt(claims: dict) -> str:
    def seg(obj):
        return base64.urlsafe_b64encode(json.dumps(obj).encode()).rstrip(b"=").decode()

    return f"{seg({'alg': 'none'})}.{seg(claims)}.sig"


def _write_auth(home, *, exp_in=3600, account_id="acct-1", auth_mode="chatgpt"):
    tokens = {"access_token": _jwt({"exp": int(time.time()) + exp_in})}
    if account_id:
        tokens["account_id"] = account_id
    home.mkdir(parents=True, exist_ok=True)
    path = home / "auth.json"
    path.write_text(json.dumps({"auth_mode": auth_mode, "tokens": tokens}))
    return path


@pytest.fixture
def codex_home(tmp_path, monkeypatch):
    home = tmp_path / "codex"
    monkeypatch.setenv("CODEX_HOME", str(home))
    llm._chatgpt_clients.clear()
    return home


def _chatgpt_cfg(**overrides) -> Config:
    return Config(
        **{"llm_provider": "chatgpt", "chat_model": "gpt-6-luna", **overrides}
    )


def _stream(text="Hypothetical passage.", input_tokens=40, output_tokens=12):
    half = len(text) // 2
    return iter(
        [
            SimpleNamespace(type="response.created"),
            SimpleNamespace(type="response.output_text.delta", delta=text[:half]),
            SimpleNamespace(type="response.output_text.delta", delta=text[half:]),
            SimpleNamespace(type="response.output_text.done", text=text),
            SimpleNamespace(
                type="response.completed",
                response=SimpleNamespace(
                    usage=SimpleNamespace(
                        input_tokens=input_tokens, output_tokens=output_tokens
                    )
                ),
            ),
        ]
    )


class TestChatGPTCredentials:
    def test_reads_codex_home_auth_file(self, codex_home):
        path = _write_auth(codex_home)
        before = path.read_text()
        token, account = llm.load_chatgpt_credentials()
        assert token == json.loads(before)["tokens"]["access_token"]
        assert account == "acct-1"
        assert path.read_text() == before  # read-only: never refreshed/rewritten

    def test_missing_file_tells_user_to_login(self, codex_home):
        with pytest.raises(KBError, match="codex login"):
            llm.load_chatgpt_credentials()

    def test_expired_token_refused(self, codex_home):
        _write_auth(codex_home, exp_in=-10)
        with pytest.raises(KBError, match="expired.*codex login"):
            llm.load_chatgpt_credentials()

    def test_token_about_to_expire_counts_as_expired(self, codex_home):
        _write_auth(codex_home, exp_in=llm.EXPIRY_SKEW_S - 1)
        with pytest.raises(KBError, match="expired"):
            llm.load_chatgpt_credentials()

    def test_api_key_login_is_not_a_chatgpt_login(self, codex_home):
        _write_auth(codex_home, auth_mode="apikey")
        with pytest.raises(KBError, match="not a ChatGPT login"):
            llm.load_chatgpt_credentials()

    def test_account_id_falls_back_to_jwt_claim(self, codex_home):
        codex_home.mkdir(parents=True)
        token = _jwt(
            {
                "exp": int(time.time()) + 3600,
                "https://api.openai.com/auth": {"chatgpt_account_id": "acct-jwt"},
            }
        )
        (codex_home / "auth.json").write_text(
            json.dumps({"auth_mode": "chatgpt", "tokens": {"access_token": token}})
        )
        assert llm.load_chatgpt_credentials() == (token, "acct-jwt")


class TestComplete:
    def test_chatgpt_provider_streams_responses_api(self, codex_home):
        _write_auth(codex_home)
        fake = MagicMock()
        fake.responses.create.return_value = _stream("Answer text")
        with patch("kb.llm.OpenAI", return_value=fake) as ctor:
            text, prompt, completion = llm.complete(
                _chatgpt_cfg(llm_reasoning_effort="medium"),
                None,
                model="gpt-6-luna",
                system="Be terse.",
                user="What is kb?",
                temperature=0.3,
                max_tokens=50,
            )

        assert (text, prompt, completion) == ("Answer text", 40, 12)
        client_kwargs = ctor.call_args.kwargs
        assert client_kwargs["base_url"] == llm.CHATGPT_BASE_URL
        assert client_kwargs["default_headers"] == {
            "chatgpt-account-id": "acct-1",
            "originator": "codex_cli_rs",
        }
        req = fake.responses.create.call_args.kwargs
        assert req["stream"] is True
        assert req["store"] is False
        assert req["instructions"] == "Be terse."
        assert req["input"] == [
            {"role": "user", "content": [{"type": "input_text", "text": "What is kb?"}]}
        ]
        assert req["reasoning"] == {"effort": "medium"}
        assert "temperature" not in req  # fixed by the Codex backend

    def test_expired_login_fails_before_any_request(self, codex_home):
        _write_auth(codex_home, exp_in=-10)
        with patch("kb.llm.OpenAI") as ctor:
            with pytest.raises(KBError, match="codex login"):
                llm.complete(
                    _chatgpt_cfg(),
                    None,
                    model="gpt-6-luna",
                    system="s",
                    user="u",
                    temperature=0.3,
                    max_tokens=10,
                )
        ctor.assert_not_called()

    def test_openai_provider_uses_chat_completions(self):
        client = MagicMock()
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
            usage=SimpleNamespace(prompt_tokens=7, completion_tokens=2),
        )
        result = llm.complete(
            Config(),
            client,
            model="gpt-4o-mini",
            system="",
            user="hi",
            temperature=0.7,
            max_tokens=20,
            json_mode=True,
        )
        assert result == ("ok", 7, 2)
        req = client.chat.completions.create.call_args.kwargs
        assert req["messages"] == [{"role": "user", "content": "hi"}]
        assert req["response_format"] == {"type": "json_object"}
        assert req["max_tokens"] == 20

    @pytest.mark.parametrize(
        ("effort", "has_temperature"), [("none", True), ("low", False)]
    )
    def test_openai_reasoning_model_params(self, effort, has_temperature):
        client = MagicMock()
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
            usage=None,
        )
        llm.complete(
            Config(llm_reasoning_effort=effort),
            client,
            model="gpt-6-luna",
            system="",
            user="hi",
            temperature=0.3,
            max_tokens=20,
        )
        req = client.chat.completions.create.call_args.kwargs
        assert "max_tokens" not in req
        assert req["max_completion_tokens"] == 20
        assert req["reasoning_effort"] == effort
        assert ("temperature" in req) is has_temperature

    def test_unknown_provider_rejected(self):
        with pytest.raises(KBError, match="llm_provider"):
            llm.complete(
                Config(llm_provider="bogus"),
                MagicMock(),
                model="m",
                system="s",
                user="u",
                temperature=0,
                max_tokens=1,
            )


class TestRouting:
    def test_hyde_base_url_override_stays_on_openai(self):
        cfg = _chatgpt_cfg(hyde_base_url="http://localhost:11434/v1")
        assert llm.hyde_provider(cfg) == llm.OPENAI
        assert llm.hyde_provider(_chatgpt_cfg()) == llm.CHATGPT

    def test_api_key_client_only_needed_for_openai_work(self):
        assert not llm.openai_client_needed(_chatgpt_cfg(embed_method="local"))
        assert llm.openai_client_needed(_chatgpt_cfg(embed_method="openai"))
        assert llm.openai_client_needed(Config(embed_method="local"))

    @pytest.mark.parametrize(
        ("overrides", "include_answer", "needed"),
        [
            ({}, False, False),
            ({}, True, True),
            ({"llm_provider": "chatgpt"}, True, False),
            ({"embed_method": "openai"}, False, True),
            ({"hyde_method": "llm"}, False, True),
            ({"hyde_method": "llm", "hyde_enabled": False}, False, False),
            (
                {"hyde_method": "llm", "hyde_base_url": "http://localhost/v1"},
                False,
                False,
            ),
            ({"expand_method": "llm"}, False, True),
            ({"expand_method": "llm", "query_expand": False}, False, False),
            ({"rerank_method": "llm"}, False, False),
        ],
    )
    def test_local_client_requirements(self, overrides, include_answer, needed):
        cfg = Config(
            **{
                "embed_method": "local",
                "hyde_method": "local",
                "expand_method": "local",
                "query_expand": True,
                "rerank_method": "cross-encoder",
                **overrides,
            }
        )
        assert llm.openai_client_needed(cfg, include_answer=include_answer) is needed

    def test_chatgpt_hyde_costs_zero_but_keeps_tokens(self, codex_home):
        _write_auth(codex_home)
        fake = MagicMock()
        fake.responses.create.return_value = _stream("A passage.", 55, 21)
        with patch("kb.llm.OpenAI", return_value=fake):
            passage, _ms, usage = llm_hyde_passage_with_usage(
                "query", None, _chatgpt_cfg()
            )

        assert passage == "A passage."
        item = _chat_cost_item(name="hyde", **usage)
        assert item["provider"] == "chatgpt"
        assert (item["prompt_tokens"], item["completion_tokens"]) == (55, 21)
        assert item["usd"] == 0.0
        # gpt-6-luna has no API price, yet the total stays known.
        assert cost_summary([item])["known"] is True

    def test_hyde_surfaces_login_errors_instead_of_degrading(self, codex_home):
        with pytest.raises(KBError, match="codex login"):
            llm_hyde_passage_with_usage("query", None, _chatgpt_cfg())
