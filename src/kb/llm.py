"""Chat completion helper shared by HyDE, query expansion, LLM rerank, and ask.

Two providers:

- ``openai``: OpenAI Chat Completions with an API key (or any OpenAI-compatible
  endpoint via the passed client).
- ``chatgpt``: the ChatGPT subscription's Codex backend (Responses API), using
  the OAuth tokens that ``codex login`` stores in ``$CODEX_HOME/auth.json``.
  The file is only read, never refreshed or written: rotating the refresh token
  here would log the Codex CLI out.
"""

from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path

from openai import AuthenticationError, OpenAI, PermissionDeniedError

from .config import Config
from .cost import usage_tokens

OPENAI = "openai"
CHATGPT = "chatgpt"
LLM_PROVIDERS = (OPENAI, CHATGPT)

CHATGPT_BASE_URL = "https://chatgpt.com/backend-api/codex"
CHATGPT_ORIGINATOR = "codex_cli_rs"
# Treat tokens this close to expiry as expired so a call never races the deadline.
EXPIRY_SKEW_S = 60
# The Codex backend requires non-empty instructions.
_DEFAULT_INSTRUCTIONS = "Follow the user's instructions exactly."

_LOGIN_HINT = "Run `codex login` (or any codex command to refresh the session)."

_chatgpt_clients: dict[tuple[str, str], OpenAI] = {}

# OpenAI reasoning families reject `max_tokens` and take `reasoning_effort`;
# they only accept `temperature` when reasoning effort is "none".
_REASONING_MODEL_PREFIXES = ("gpt-5", "gpt-6", "o1", "o3", "o4")


def _kb_error(message: str) -> Exception:
    from .api import KBError  # api imports this module's callers; avoid a cycle

    return KBError(message)


def codex_auth_path() -> Path:
    home = os.environ.get("CODEX_HOME") or str(Path.home() / ".codex")
    return Path(home).expanduser() / "auth.json"


def _jwt_claims(token: str) -> dict:
    try:
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        return json.loads(base64.urlsafe_b64decode(payload))
    except Exception:
        return {}


def load_chatgpt_credentials(now: float | None = None) -> tuple[str, str]:
    """Return (access_token, account_id) from the Codex auth file (read-only)."""
    path = codex_auth_path()
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError:
        raise _kb_error(
            f"ChatGPT credentials not found at {path}. {_LOGIN_HINT}"
        ) from None
    except (OSError, ValueError) as exc:
        raise _kb_error(f"Cannot read ChatGPT credentials at {path}: {exc}") from None

    if data.get("auth_mode") != "chatgpt":
        raise _kb_error(
            f"{path} is not a ChatGPT login (auth_mode={data.get('auth_mode')!r}). "
            + _LOGIN_HINT
        )
    tokens = data.get("tokens") or {}
    access_token = tokens.get("access_token")
    if not access_token:
        raise _kb_error(f"No ChatGPT access token in {path}. {_LOGIN_HINT}")

    claims = _jwt_claims(access_token)
    exp = claims.get("exp")
    if not isinstance(exp, (int, float)):
        raise _kb_error(f"Unreadable ChatGPT access token in {path}. {_LOGIN_HINT}")
    if exp <= (time.time() if now is None else now) + EXPIRY_SKEW_S:
        raise _kb_error(f"ChatGPT access token in {path} has expired. {_LOGIN_HINT}")

    account_id = tokens.get("account_id") or (
        claims.get("https://api.openai.com/auth") or {}
    ).get("chatgpt_account_id")
    if not account_id:
        raise _kb_error(f"No ChatGPT account id in {path}. {_LOGIN_HINT}")
    return access_token, account_id


def hyde_provider(cfg: Config) -> str:
    """HyDE's provider: a dedicated hyde_base_url endpoint is always OpenAI-compatible."""
    return OPENAI if cfg.hyde_base_url else cfg.llm_provider


def openai_client_needed(cfg: Config) -> bool:
    """Whether a default OpenAI API-key client is needed (embeddings or LLM calls)."""
    return cfg.embed_method != "local" or cfg.llm_provider != CHATGPT


def _chatgpt_client() -> OpenAI:
    access_token, account_id = load_chatgpt_credentials()
    key = (access_token, account_id)
    client = _chatgpt_clients.get(key)
    if client is None:
        _chatgpt_clients.clear()
        client = OpenAI(
            base_url=CHATGPT_BASE_URL,
            api_key=access_token,
            default_headers={
                "chatgpt-account-id": account_id,
                "originator": CHATGPT_ORIGINATOR,
            },
        )
        _chatgpt_clients[key] = client
    return client


def _complete_chatgpt(
    cfg: Config, model: str, system: str, user: str, json_mode: bool
) -> tuple[str, int | None, int | None]:
    # The Codex backend fixes temperature, and max_output_tokens would also cap
    # reasoning tokens, so neither is sent; output length follows the prompt.
    kwargs: dict = {
        "model": model,
        "instructions": system or _DEFAULT_INSTRUCTIONS,
        "input": [{"role": "user", "content": [{"type": "input_text", "text": user}]}],
        "reasoning": {"effort": cfg.llm_reasoning_effort},
        "store": False,
        "stream": True,
    }
    if json_mode:
        kwargs["text"] = {"format": {"type": "json_object"}}

    try:
        stream = _chatgpt_client().responses.create(**kwargs)
        deltas: list[str] = []
        done_texts: list[str] = []
        usage = None
        for event in stream:
            etype = getattr(event, "type", "")
            if etype == "response.output_text.delta":
                deltas.append(event.delta)
            elif etype == "response.output_text.done":
                done_texts.append(event.text)
            elif etype == "response.completed":
                usage = getattr(event.response, "usage", None)
            elif etype in ("response.failed", "response.incomplete", "error"):
                response = getattr(event, "response", None)
                detail = getattr(response, "error", None) or getattr(
                    event, "message", etype
                )
                raise RuntimeError(f"ChatGPT request failed: {detail}")
    except (AuthenticationError, PermissionDeniedError) as exc:
        raise _kb_error(
            f"ChatGPT rejected the Codex login ({exc.status_code}). {_LOGIN_HINT}"
        ) from exc

    text = "".join(done_texts) or "".join(deltas)
    prompt = getattr(usage, "input_tokens", None)
    completion = getattr(usage, "output_tokens", None)
    return (
        text,
        prompt if isinstance(prompt, int) else None,
        completion if isinstance(completion, int) else None,
    )


def complete(
    cfg: Config,
    client: OpenAI | None,
    *,
    model: str,
    system: str,
    user: str,
    temperature: float,
    max_tokens: int,
    json_mode: bool = False,
    provider: str | None = None,
) -> tuple[str, int | None, int | None]:
    """Run one chat completion; return (text, prompt_tokens, completion_tokens).

    Token counts are None when the provider did not report usage. ``provider``
    defaults to ``cfg.llm_provider``; ``client`` is used only by the openai provider.
    """
    provider = provider or cfg.llm_provider
    if provider == CHATGPT:
        return _complete_chatgpt(cfg, model, system, user, json_mode)
    if provider != OPENAI:
        raise _kb_error(
            f"Unknown llm_provider {provider!r}; use one of: {', '.join(LLM_PROVIDERS)}"
        )

    messages = [{"role": "system", "content": system}] if system else []
    messages.append({"role": "user", "content": user})
    kwargs: dict = {"model": model, "messages": messages}
    if model.startswith(_REASONING_MODEL_PREFIXES):
        effort = cfg.llm_reasoning_effort
        kwargs["reasoning_effort"] = effort
        kwargs["max_completion_tokens"] = max_tokens
        if effort == "none":
            kwargs["temperature"] = temperature
    else:
        kwargs["temperature"] = temperature
        kwargs["max_tokens"] = max_tokens
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    resp = client.chat.completions.create(**kwargs)
    prompt_tokens, completion_tokens = usage_tokens(getattr(resp, "usage", None))
    return (resp.choices[0].message.content or ""), prompt_tokens, completion_tokens
