"""Query expansion: generate keyword synonyms (lex) and semantic rephrasings (vec).

Two methods:
- local: FLAN-T5-small via transformers (no API cost, ~1s on CPU)
- llm: OpenAI API call with JSON mode
"""

import json
import logging
import time

from openai import OpenAI

from .config import Config

log = logging.getLogger(__name__)

# Lazy-loaded T5 model cache (same pattern as rerank.py cross-encoder)
_expand_model_cache: dict[str, tuple] = {}

_LLM_PROMPT = """\
Generate search query expansions. Return JSON: {"lex": ["keyword variant 1", ...], "vec": ["semantic rephrasing 1", ...]}
Rules:
- lex: 2-3 short keyword alternatives (synonyms, related terms)
- vec: 1-2 natural language rephrasings (different wording, same intent)
- Do not repeat the original query
Query: "{query}"\
"""


def expand_query(
    client: OpenAI, query: str, cfg: Config
) -> tuple[list[dict], float]:
    """Dispatch to configured method. Returns ([{"type": "lex"|"vec", "text": "..."}], elapsed_ms)."""
    t0 = time.time()
    if cfg.expand_method == "llm":
        result = llm_expand(client, query, cfg)
    else:
        result = local_expand(query, cfg)
    elapsed = (time.time() - t0) * 1000
    return result, elapsed


def llm_expand(client: OpenAI, query: str, cfg: Config) -> list[dict]:
    """OpenAI API expansion with JSON mode."""
    try:
        resp = client.chat.completions.create(
            model=cfg.chat_model,
            messages=[
                {"role": "user", "content": _LLM_PROMPT.format(query=query)},
            ],
            response_format={"type": "json_object"},
            temperature=0.7,
            max_tokens=200,
        )
        text = (resp.choices[0].message.content or "").strip()
        data = json.loads(text)
    except Exception:
        log.warning("Query expansion (LLM) failed", exc_info=True)
        return []

    results = []
    query_lower = query.lower().strip()
    for variant in data.get("lex", []):
        if isinstance(variant, str) and variant.lower().strip() != query_lower:
            results.append({"type": "lex", "text": variant})
    for variant in data.get("vec", []):
        if isinstance(variant, str) and variant.lower().strip() != query_lower:
            results.append({"type": "vec", "text": variant})
    return results


def _get_t5_model(model_name: str):
    """Load and cache a T5 model + tokenizer (lazy import)."""
    if model_name not in _expand_model_cache:
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError:
            raise ImportError(
                "transformers is required for local query expansion. "
                "Install with: pip install 'kb[expand]' or pip install transformers"
            )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        _expand_model_cache[model_name] = (tokenizer, model)
    return _expand_model_cache[model_name]


def local_expand(query: str, cfg: Config) -> list[dict]:
    """FLAN-T5-small expansion via transformers (lazy-loaded, cached)."""
    try:
        tokenizer, model = _get_t5_model(cfg.expand_model)
    except ImportError:
        raise
    except Exception:
        log.warning("Query expansion (local) failed to load model", exc_info=True)
        return []

    results = []
    query_lower = query.lower().strip()

    try:
        # Generate keyword synonyms
        lex_prompt = f"Generate keyword synonyms for this search query: {query}"
        inputs = tokenizer(lex_prompt, return_tensors="pt", truncation=True)
        outputs = model.generate(**inputs, max_new_tokens=50, num_beams=3)
        lex_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        for line in lex_text.replace(",", "\n").split("\n"):
            variant = line.strip()
            if variant and variant.lower() != query_lower:
                results.append({"type": "lex", "text": variant})

        # Generate semantic rephrasing
        vec_prompt = f"Rephrase this search query: {query}"
        inputs = tokenizer(vec_prompt, return_tensors="pt", truncation=True)
        outputs = model.generate(**inputs, max_new_tokens=60, num_beams=3)
        vec_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        if vec_text and vec_text.lower() != query_lower:
            results.append({"type": "vec", "text": vec_text})
    except Exception:
        log.warning("Query expansion (local) generation failed", exc_info=True)
        return []

    return results
