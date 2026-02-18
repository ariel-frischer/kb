"""Embedding helpers."""

from __future__ import annotations

import struct

from openai import OpenAI

from .config import Config

# Lazy-loaded SentenceTransformer model cache (same pattern as rerank.py)
_embed_model_cache: dict[str, object] = {}


def serialize_f32(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def deserialize_f32(blob: bytes) -> list[float]:
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))


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


def _get_embed_model(model_name: str):
    """Load and cache a SentenceTransformer model, using GPU if available."""
    if model_name not in _embed_model_cache:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for local embeddings. "
                "Install it with: pip install 'kb[local-embed]' or pip install sentence-transformers"
            )
        device = _get_device()
        _embed_model_cache[model_name] = SentenceTransformer(model_name, device=device)
    return _embed_model_cache[model_name]


def local_embed_batch(
    texts: list[str], cfg: Config, *, is_query: bool = False
) -> list[list[float]]:
    """Embed texts using a local SentenceTransformer model.

    Passes prompt_name="query" for query embeddings only when the model
    declares prompt templates (e.g. arctic-embed). Models without prompts
    (e.g. Granite R2) get plain encode().
    Truncates to cfg.embed_dims if set lower than the model's native output.
    """
    model = _get_embed_model(cfg.local_embed_model)
    kwargs: dict = {"normalize_embeddings": True}
    if is_query and getattr(model, "prompts", None):
        kwargs["prompt_name"] = "query"
    embeddings = model.encode(texts, **kwargs)

    result = [emb.tolist() for emb in embeddings]

    # Matryoshka truncation if embed_dims < model output
    native_dims = len(result[0]) if result else 0
    if native_dims and cfg.embed_dims < native_dims:
        result = [emb[: cfg.embed_dims] for emb in result]

    return result


def local_embed_dims(cfg: Config) -> int:
    """Return effective embedding dimensions for a local model.

    Auto-detects native model dims. Returns min(cfg.embed_dims, native) to
    support Matryoshka truncation while preventing dimension mismatches
    (e.g. default 1536 vs model's 768).
    """
    model = _get_embed_model(cfg.local_embed_model)
    native = model.get_sentence_embedding_dimension()
    return min(cfg.embed_dims, native)


def embed_batch(
    client: OpenAI | None,
    texts: list[str],
    cfg: Config,
    *,
    is_query: bool = False,
) -> list[list[float]]:
    """Embed a batch of texts using the configured method.

    Dispatches to local_embed_batch() or OpenAI API based on cfg.embed_method.
    """
    if cfg.embed_method == "local":
        return local_embed_batch(texts, cfg, is_query=is_query)

    resp = client.embeddings.create(
        model=cfg.embed_model, input=texts, dimensions=cfg.embed_dims
    )
    return [d.embedding for d in resp.data]
