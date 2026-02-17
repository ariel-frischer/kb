"""Reranking: RankGPT (LLM) or local cross-encoder."""

import re
import time

from openai import OpenAI

from .config import Config

# Lazy-loaded cross-encoder model cache
_cross_encoder_cache: dict[str, object] = {}


def rerank(
    client: OpenAI | None,
    question: str,
    results: list[dict],
    cfg: Config,
) -> tuple[list[dict], dict]:
    """Dispatch to the configured rerank method."""
    if cfg.rerank_method == "cross-encoder":
        return cross_encoder_rerank(question, results, cfg)
    return llm_rerank(client, question, results, cfg)


def llm_rerank(
    client: OpenAI,
    question: str,
    results: list[dict],
    cfg: Config,
) -> tuple[list[dict], dict]:
    """RankGPT-style reranking: present numbered passages, ask LLM for ranking."""
    if len(results) <= cfg.rerank_top_k:
        return results, {}

    passages = []
    for i, r in enumerate(results):
        text = (r.get("text") or "")[:500]
        source = r.get("doc_path") or "unknown"
        heading = r.get("heading") or ""
        label = source
        if heading:
            label += f" > {heading}"
        passages.append(f"[{i + 1}] ({label})\n{text}")

    passages_text = "\n\n".join(passages)

    t0 = time.time()
    resp = client.chat.completions.create(
        model=cfg.chat_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a relevance ranking assistant. Given a question and numbered passages, "
                    "rank the passages by relevance to the question. Output ONLY a comma-separated "
                    "list of passage numbers from most to least relevant. Example: 3,7,1,5,2,4,6"
                ),
            },
            {
                "role": "user",
                "content": f"Question: {question}\n\nPassages:\n{passages_text}\n\nRanking:",
            },
        ],
        temperature=0,
        max_tokens=200,
    )
    ranking_text = resp.choices[0].message.content.strip()
    rerank_ms = (time.time() - t0) * 1000
    tokens = resp.usage

    ranked_indices = []
    for num_str in re.findall(r"\d+", ranking_text):
        idx = int(num_str) - 1
        if 0 <= idx < len(results) and idx not in ranked_indices:
            ranked_indices.append(idx)

    # Append any the LLM missed
    for i in range(len(results)):
        if i not in ranked_indices:
            ranked_indices.append(i)

    reranked = [results[i] for i in ranked_indices[: cfg.rerank_top_k]]

    rerank_info = {
        "rerank_ms": rerank_ms,
        "prompt_tokens": tokens.prompt_tokens,
        "completion_tokens": tokens.completion_tokens,
        "input_count": len(results),
        "output_count": len(reranked),
    }

    return reranked, rerank_info


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


def _get_cross_encoder(model_name: str):
    """Load and cache a cross-encoder model (lazy import), using GPU if available."""
    if model_name not in _cross_encoder_cache:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for cross-encoder reranking. "
                "Install it with: pip install 'kb[rerank]' or pip install sentence-transformers"
            )
        device = _get_device()
        _cross_encoder_cache[model_name] = CrossEncoder(model_name, device=device)
    return _cross_encoder_cache[model_name]


def cross_encoder_rerank(
    question: str,
    results: list[dict],
    cfg: Config,
) -> tuple[list[dict], dict]:
    """Rerank using a local cross-encoder model."""
    if len(results) <= cfg.rerank_top_k:
        return results, {}

    model = _get_cross_encoder(cfg.cross_encoder_model)

    pairs = []
    for r in results:
        text = (r.get("text") or "")[:500]
        heading = r.get("heading") or ""
        passage = f"{heading}\n{text}".strip() if heading else text
        pairs.append((question, passage))

    t0 = time.time()
    scores = model.predict(pairs)
    rerank_ms = (time.time() - t0) * 1000

    scored = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    reranked = [results[i] for i, _ in scored[: cfg.rerank_top_k]]

    rerank_info = {
        "rerank_ms": rerank_ms,
        "model": cfg.cross_encoder_model,
        "input_count": len(results),
        "output_count": len(reranked),
    }

    return reranked, rerank_info
