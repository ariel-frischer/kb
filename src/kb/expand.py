"""Query expansion: generate keyword synonyms (lex) and semantic rephrasings (vec).

Two methods:
- local: small causal LM (default Qwen3-0.6B) via transformers, no API cost
- llm: OpenAI API call with JSON mode
"""

import json
import logging
import re
import time

from openai import OpenAI

from .config import Config
from .cost import estimate_tokens
from .hyde import load_causal_lm
from .llm import complete

log = logging.getLogger(__name__)

_LLM_PROMPT = """\
Generate search query expansions. Return JSON: {{"lex": ["keyword variant 1", ...], "vec": ["semantic rephrasing 1", ...]}}
Rules:
- lex: 2-3 short keyword alternatives (synonyms, related terms)
- vec: 1-2 natural language rephrasings (different wording, same intent)
- Do not repeat the original query
Query: "{query}"\
"""
EXPAND_MAX_TOKENS = 200


def expand_query(
    client: OpenAI | None, query: str, cfg: Config
) -> tuple[list[dict], float]:
    """Dispatch to configured method. Returns ([{"type": "lex"|"vec", "text": "..."}], elapsed_ms)."""
    t0 = time.time()
    if cfg.expand_method == "llm":
        result = llm_expand(client, query, cfg)
    else:
        result = local_expand(query, cfg)
    elapsed = (time.time() - t0) * 1000
    return result, elapsed


def expand_query_with_usage(
    client: OpenAI | None, query: str, cfg: Config
) -> tuple[list[dict], float, dict | None]:
    """Like expand_query, plus LLM token usage metadata (None for local expansion)."""
    t0 = time.time()
    usage = None
    if cfg.expand_method == "llm":
        result, usage = llm_expand_with_usage(client, query, cfg)
    else:
        result = local_expand(query, cfg)
    elapsed = (time.time() - t0) * 1000
    return result, elapsed, usage


def llm_expand(client: OpenAI | None, query: str, cfg: Config) -> list[dict]:
    """LLM expansion with JSON mode."""
    return llm_expand_with_usage(client, query, cfg)[0]


def llm_expand_with_usage(
    client: OpenAI | None, query: str, cfg: Config
) -> tuple[list[dict], dict | None]:
    """LLM expansion plus token usage (None when the request failed)."""
    from .api import KBError  # api imports expand; import lazily to avoid a cycle

    prompt = _LLM_PROMPT.format(query=query)
    try:
        text, prompt_tokens, completion_tokens = complete(
            cfg,
            client,
            model=cfg.chat_model,
            system="",
            user=prompt,
            temperature=0.7,
            max_tokens=EXPAND_MAX_TOKENS,
            json_mode=True,
        )
        text = text.strip()
    except KBError:
        raise
    except Exception:
        log.warning("Query expansion (LLM) failed", exc_info=True)
        return [], None

    estimated = False
    if prompt_tokens is None or completion_tokens is None:
        prompt_tokens = estimate_tokens(prompt)
        completion_tokens = estimate_tokens(text)
        estimated = True
    usage = {
        "model": cfg.chat_model,
        "provider": cfg.llm_provider,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "estimated_tokens": estimated,
    }

    return _parse_expansions(text, query), usage


def _parse_expansions(text: str, query: str) -> list[dict]:
    """Parse the {"lex": [...], "vec": [...]} contract, dropping echoes of the query."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    try:
        data = json.loads(match.group(0) if match else text)
    except Exception:
        log.warning("Query expansion returned non-JSON output", exc_info=True)
        return []
    if not isinstance(data, dict):
        return []

    results = []
    query_lower = query.lower().strip()
    for kind in ("lex", "vec"):
        for variant in data.get(kind) or []:
            if isinstance(variant, str) and variant.lower().strip() != query_lower:
                results.append({"type": kind, "text": variant.strip()})
    return results


def local_expand(query: str, cfg: Config) -> list[dict]:
    """Expansion with a local chat LM (lazy-loaded, cached), same JSON contract as llm."""
    try:
        tokenizer, model, device = load_causal_lm(cfg.expand_model)
    except ImportError:
        raise
    except Exception:
        log.warning("Query expansion (local) failed to load model", exc_info=True)
        return []

    try:
        messages = [{"role": "user", "content": _LLM_PROMPT.format(query=query)}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_len = inputs["input_ids"].shape[1]
        outputs = model.generate(
            **inputs, max_new_tokens=EXPAND_MAX_TOKENS, do_sample=False
        )
        text = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    except Exception:
        log.warning("Query expansion (local) generation failed", exc_info=True)
        return []

    return _parse_expansions(text, query)
