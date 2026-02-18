"""MCP server exposing kb search/ask as tools for AI agents."""

import sys

from mcp.server.fastmcp import FastMCP

from .api import (
    KBError,
    ask_core,
    feedback_core,
    fts_core,
    list_core,
    search_core,
    similar_core,
    stats_core,
)
from .config import Config, find_config, load_secrets

mcp = FastMCP(
    "kb",
    instructions=(
        "Knowledge base tools. Search indexed documents using hybrid "
        "semantic + keyword search, ask questions with RAG, or browse the index."
    ),
)


def _get_config():
    load_secrets()
    return find_config()


def _with_expand(cfg: Config, expand: bool) -> Config:
    """Return a config copy with query_expand set if requested."""
    if expand and not cfg.query_expand:
        from copy import copy

        cfg = copy(cfg)
        cfg.query_expand = True
    return cfg


@mcp.tool()
def kb_search(
    query: str,
    top_k: int = 5,
    threshold: float | None = None,
    expand: bool = False,
) -> dict:
    """Hybrid semantic + keyword search over the knowledge base.

    Uses HyDE (Hypothetical Document Embeddings) by default: generates a hypothetical
    answer passage via LLM, embeds that for vector search. FTS uses original query.

    Supports inline filters in the query string:
      file:glob, type:name, tag:name, dt>"date", dt<"date", +"must", -"exclude"

    Args:
        query: Search query (may include inline filters).
        top_k: Number of results to return.
        threshold: Minimum similarity score (0-1). Omit to use config default.
        expand: Enable query expansion (keyword synonyms + semantic rephrasings).
    """
    try:
        cfg = _with_expand(_get_config(), expand)
        return search_core(query, cfg, top_k, threshold)
    except KBError as e:
        return {"error": str(e)}


@mcp.tool()
def kb_ask(
    question: str,
    top_k: int = 8,
    threshold: float | None = None,
    expand: bool = False,
) -> dict:
    """Ask a question and get a RAG-generated answer with source citations.

    Full pipeline: [HyDE] -> hybrid search -> LLM rerank -> confidence filter -> LLM answer.
    HyDE generates a hypothetical passage for better vector retrieval (skipped on BM25 shortcut).
    Supports the same inline filters as kb_search.

    Args:
        question: The question to answer.
        top_k: Number of context chunks to retrieve.
        threshold: Minimum similarity for context chunks.
        expand: Enable query expansion (keyword synonyms + semantic rephrasings).
    """
    try:
        cfg = _with_expand(_get_config(), expand)
        return ask_core(question, cfg, top_k, threshold)
    except KBError as e:
        return {"error": str(e)}


@mcp.tool()
def kb_fts(query: str, top_k: int = 5) -> dict:
    """Keyword-only search (FTS5). No embedding API call, instant results.

    Best for exact keyword lookups or when you want zero API cost.

    Args:
        query: Search query (may include inline filters).
        top_k: Number of results to return.
    """
    try:
        return fts_core(query, _get_config(), top_k)
    except KBError as e:
        return {"error": str(e)}


@mcp.tool()
def kb_similar(file_path: str, top_k: int = 10) -> dict:
    """Find documents similar to a given file. No API call needed.

    Args:
        file_path: Path to the source document (as indexed, or suffix match).
        top_k: Number of similar documents to return.
    """
    try:
        return similar_core(file_path, _get_config(), top_k)
    except KBError as e:
        return {"error": str(e)}


@mcp.tool()
def kb_status() -> dict:
    """Show index statistics: document count, chunk count, DB size, etc."""
    return stats_core(_get_config())


@mcp.tool()
def kb_list() -> dict:
    """List all indexed documents with type, size, and chunk count."""
    return list_core(_get_config())


@mcp.tool()
def kb_feedback(
    message: str,
    tool: str = "",
    severity: str = "note",
    context: str = "",
    agent_id: str = "",
    error_trace: str = "",
) -> dict:
    """Submit feedback about kb (bug reports, suggestions, notes).

    Appends to a local YAML file for dev review. No config needed.

    Args:
        message: The feedback message (required).
        tool: Which kb tool the feedback is about (e.g. "kb_search").
        severity: One of "bug", "suggestion", or "note".
        context: Additional context (query that failed, environment info, etc.).
        agent_id: Identifier for the calling agent.
        error_trace: Error traceback or log snippet.
    """
    try:
        return feedback_core(
            message,
            tool=tool,
            severity=severity,
            context=context,
            agent_id=agent_id,
            error_trace=error_trace,
        )
    except KBError as e:
        return {"error": str(e)}


def main():
    print("kb MCP server starting (stdio transport)...", file=sys.stderr)
    mcp.run(transport="stdio")
