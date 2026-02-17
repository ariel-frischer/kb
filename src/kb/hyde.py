"""HyDE (Hypothetical Document Embeddings) — generate a hypothetical answer passage.

The passage is embedded instead of the raw query for vector search, improving retrieval
for question-style queries. FTS still uses the original query keywords.

Two methods:
- llm: OpenAI API call (current default)
- local: causal LM via transformers (no API cost)
"""

import logging
import time

from openai import OpenAI

from .config import Config

log = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "Given a search query, write a short passage (100-200 words) that would directly "
    "answer the query. Write as if from an authoritative document. No preamble."
)

# Lazy-loaded local model cache (same pattern as expand.py _expand_model_cache)
_hyde_model_cache: dict[str, tuple] = {}


def _get_device() -> str:
    """Pick the best available torch device."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _get_local_model(model_name: str):
    """Load and cache a causal LM + tokenizer (lazy import)."""
    if model_name not in _hyde_model_cache:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError(
                "transformers and torch are required for local HyDE. "
                "Install with: pip install 'kb[local-llm]' or pip install transformers torch"
            )
        device = _get_device()
        dtype = torch.bfloat16 if device != "cpu" and torch.cuda.is_bf16_supported() else torch.float32
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype
        ).to(device)
        model.eval()
        _hyde_model_cache[model_name] = (tokenizer, model, device)
    return _hyde_model_cache[model_name]


def local_hyde_passage(query: str, cfg: Config) -> tuple[str | None, float]:
    """Generate a hypothetical passage using a local causal LM."""
    t0 = time.time()
    try:
        tokenizer, model, device = _get_local_model(cfg.hyde_local_model)
    except ImportError:
        raise
    except Exception:
        elapsed = (time.time() - t0) * 1000
        log.warning("HyDE (local): failed to load model", exc_info=True)
        return None, elapsed

    try:
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(device)
        input_len = inputs["input_ids"].shape[1]

        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.7,
            do_sample=True,
        )
        passage = tokenizer.decode(
            outputs[0][input_len:], skip_special_tokens=True
        ).strip()

        elapsed = (time.time() - t0) * 1000
        if not passage:
            log.warning("HyDE (local): empty response from %s", cfg.hyde_local_model)
            return None, elapsed
        return passage, elapsed
    except Exception:
        elapsed = (time.time() - t0) * 1000
        log.warning("HyDE (local): generation failed", exc_info=True)
        return None, elapsed


def llm_hyde_passage(
    query: str, client: OpenAI, cfg: Config
) -> tuple[str | None, float]:
    """Generate a hypothetical passage via OpenAI API."""
    model = cfg.hyde_model or cfg.chat_model
    t0 = time.time()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            temperature=0.7,
            max_tokens=300,
        )
        passage = (resp.choices[0].message.content or "").strip()
        elapsed = (time.time() - t0) * 1000
        if not passage:
            log.warning("HyDE: empty response from %s", model)
            return None, elapsed
        return passage, elapsed
    except Exception:
        elapsed = (time.time() - t0) * 1000
        log.warning("HyDE: failed to generate passage", exc_info=True)
        return None, elapsed


def generate_hyde_passage(
    query: str, client: OpenAI | None, cfg: Config
) -> tuple[str | None, float]:
    """Generate a hypothetical document passage for the given query.

    Dispatches to local or LLM method based on cfg.hyde_method.
    Returns (passage, elapsed_ms). Returns (None, elapsed_ms) on failure.
    """
    if cfg.hyde_method == "local":
        return local_hyde_passage(query, cfg)
    return llm_hyde_passage(query, client, cfg)
