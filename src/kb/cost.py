"""Lightweight API cost estimation for kb commands."""

from __future__ import annotations

import math
from typing import Any


CHAT_PRICES_PER_1M = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
}

EMBED_PRICES_PER_1M = {
    "text-embedding-3-small": 0.02,
    "text-embedding-3-large": 0.13,
}


def estimate_tokens(text: str) -> int:
    """Cheap token estimate for cost display when API usage is unavailable."""
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


def usage_tokens(usage: Any) -> tuple[int | None, int | None]:
    """Extract OpenAI-style usage tokens, returning None when unavailable."""
    prompt = getattr(usage, "prompt_tokens", None)
    completion = getattr(usage, "completion_tokens", None)
    if not isinstance(prompt, int):
        prompt = None
    if not isinstance(completion, int):
        completion = None
    return prompt, completion


def _price_for_model(model: str, prices: dict[str, Any]) -> Any | None:
    for prefix, price in sorted(
        prices.items(), key=lambda item: len(item[0]), reverse=True
    ):
        if model == prefix or model.startswith(f"{prefix}-"):
            return price
    return None


def chat_cost_usd(
    model: str, prompt_tokens: int, completion_tokens: int
) -> float | None:
    price = _price_for_model(model, CHAT_PRICES_PER_1M)
    if not price:
        return None
    input_cost = prompt_tokens * price["input"] / 1_000_000
    output_cost = completion_tokens * price["output"] / 1_000_000
    return input_cost + output_cost


def embed_cost_usd(model: str, tokens: int) -> float | None:
    price = _price_for_model(model, EMBED_PRICES_PER_1M)
    if price is None:
        return None
    return tokens * price / 1_000_000


def cost_summary(items: list[dict]) -> dict:
    known_items = [item for item in items if item.get("usd") is not None]
    unknown_items = [item for item in items if item.get("usd") is None]
    total = sum(item["usd"] for item in known_items)
    return {
        "estimated_total_usd": total,
        "currency": "USD",
        "known": not unknown_items,
        "items": items,
    }


def format_usd(value: float) -> str:
    if value < 0.0001:
        return f"${value:.6f}"
    if value < 0.01:
        return f"${value:.5f}"
    return f"${value:.4f}"
