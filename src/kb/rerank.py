"""LLM reranking using the RankGPT pattern."""

import re
import time

from openai import OpenAI

from .config import Config


def llm_rerank(
    client: OpenAI,
    question: str,
    results: list[dict],
    cfg: Config,
) -> list[dict]:
    """RankGPT-style reranking: present numbered passages, ask LLM for ranking."""
    if len(results) <= cfg.rerank_top_k:
        return results

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

    print(
        f"(rerank: {rerank_ms:.0f}ms, {tokens.prompt_tokens}+{tokens.completion_tokens} tokens, "
        f"{len(results)} -> {len(reranked)})"
    )

    return reranked
