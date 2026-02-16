"""Embedding helpers."""

import struct

from openai import OpenAI

from .config import Config


def serialize_f32(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def deserialize_f32(blob: bytes) -> list[float]:
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))


def embed_batch(client: OpenAI, texts: list[str], cfg: Config) -> list[list[float]]:
    resp = client.embeddings.create(
        model=cfg.embed_model, input=texts, dimensions=cfg.embed_dims
    )
    return [d.embedding for d in resp.data]
