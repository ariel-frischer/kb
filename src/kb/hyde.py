"""HyDE (Hypothetical Document Embeddings) — generate a hypothetical answer passage via LLM.

The passage is embedded instead of the raw query for vector search, improving retrieval
for question-style queries. FTS still uses the original query keywords.
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


def generate_hyde_passage(
    query: str, client: OpenAI, cfg: Config
) -> tuple[str | None, float]:
    """Generate a hypothetical document passage for the given query.

    Returns (passage, elapsed_ms). Returns (None, elapsed_ms) on failure.
    """
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
