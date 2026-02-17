"""MCP server exposing kb search/ask as tools for AI agents."""

from mcp.server.fastmcp import FastMCP

from .api import (
    KBError,
    ask_core,
    fts_core,
    list_core,
    search_core,
    similar_core,
    stats_core,
)
from .config import find_config, load_secrets

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


@mcp.tool()
def kb_search(query: str, top_k: int = 5, threshold: float | None = None) -> dict:
    """Hybrid semantic + keyword search over the knowledge base.

    Supports inline filters in the query string:
      file:glob, type:name, tag:name, dt>"date", dt<"date", +"must", -"exclude"

    Args:
        query: Search query (may include inline filters).
        top_k: Number of results to return.
        threshold: Minimum similarity score (0-1). Omit to use config default.
    """
    try:
        return search_core(query, _get_config(), top_k, threshold)
    except KBError as e:
        return {"error": str(e)}


@mcp.tool()
def kb_ask(question: str, top_k: int = 8, threshold: float | None = None) -> dict:
    """Ask a question and get a RAG-generated answer with source citations.

    Full pipeline: hybrid search -> LLM rerank -> confidence filter -> LLM answer.
    Supports the same inline filters as kb_search.

    Args:
        question: The question to answer.
        top_k: Number of context chunks to retrieve.
        threshold: Minimum similarity for context chunks.
    """
    try:
        return ask_core(question, _get_config(), top_k, threshold)
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


def main():
    mcp.run(transport="stdio")
