"""Tests for kb.cost."""

from kb.cost import chat_cost_usd, cost_summary, embed_cost_usd, estimate_tokens


def test_estimate_tokens_from_chars():
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abcde") == 2


def test_chat_cost_matches_default_model_rates():
    cost = chat_cost_usd(
        "gpt-4o-mini", prompt_tokens=1_000_000, completion_tokens=1_000_000
    )
    assert cost == 0.75


def test_chat_cost_matches_snapshot_prefix():
    cost = chat_cost_usd(
        "gpt-4o-mini-2024-07-18",
        prompt_tokens=1_000_000,
        completion_tokens=1_000_000,
    )
    assert cost == 0.75


def test_embedding_cost_matches_default_model_rate():
    assert embed_cost_usd("text-embedding-3-small", 1_000_000) == 0.02


def test_cost_summary_marks_unknown_price_items():
    summary = cost_summary(
        [
            {"name": "answer", "usd": 0.001},
            {"name": "custom-provider", "usd": None},
        ]
    )

    assert summary["estimated_total_usd"] == 0.001
    assert summary["known"] is False
