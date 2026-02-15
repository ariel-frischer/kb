"""Pre-search filter parsing and application."""

import re
import sqlite3
from fnmatch import fnmatch


def parse_filters(query: str) -> tuple[str, dict]:
    """Parse inline filter syntax from query, return (clean_query, filters).

    Supported filters:
      file:pattern       — glob on doc_path
      dt>"YYYY-MM-DD"    — indexed after date
      dt<"YYYY-MM-DD"    — indexed before date
      +"keyword"         — chunk must contain keyword
      -"keyword"         — chunk must not contain keyword
    """
    filters: dict = {
        "file_glob": None,
        "date_after": None,
        "date_before": None,
        "must_contain": [],
        "must_not_contain": [],
    }

    clean = query

    m = re.search(r"file:(\S+)", clean)
    if m:
        filters["file_glob"] = m.group(1)
        clean = clean[: m.start()] + clean[m.end() :]

    m = re.search(r'dt>=?"(\d{4}-\d{2}-\d{2})"', clean)
    if m:
        filters["date_after"] = m.group(1)
        clean = clean[: m.start()] + clean[m.end() :]

    m = re.search(r'dt<=?"(\d{4}-\d{2}-\d{2})"', clean)
    if m:
        filters["date_before"] = m.group(1)
        clean = clean[: m.start()] + clean[m.end() :]

    for m in re.finditer(r'\+"([^"]+)"', clean):
        filters["must_contain"].append(m.group(1).lower())
    clean = re.sub(r'\+"[^"]*"', "", clean)

    for m in re.finditer(r'-"([^"]+)"', clean):
        filters["must_not_contain"].append(m.group(1).lower())
    clean = re.sub(r'-"[^"]*"', "", clean)

    clean = re.sub(r"\s+", " ", clean).strip()
    return clean, filters


def apply_filters(
    results: list[dict], filters: dict, conn: sqlite3.Connection
) -> list[dict]:
    """Apply parsed filters to search results."""
    if not any(filters.values()):
        return results

    dates_cache: dict[str, str] = {}
    if filters["date_after"] or filters["date_before"]:
        rows = conn.execute("SELECT path, indexed_at FROM documents").fetchall()
        dates_cache = {r["path"]: r["indexed_at"] for r in rows}

    filtered = []
    for r in results:
        doc_path = r.get("doc_path") or ""
        text_lower = (r.get("text") or "").lower()

        if filters["file_glob"] and not fnmatch(doc_path, filters["file_glob"]):
            continue

        if filters["date_after"]:
            indexed = dates_cache.get(doc_path, "")
            if indexed < filters["date_after"]:
                continue
        if filters["date_before"]:
            indexed = dates_cache.get(doc_path, "")
            if indexed > filters["date_before"]:
                continue

        if any(kw not in text_lower for kw in filters["must_contain"]):
            continue
        if any(kw in text_lower for kw in filters["must_not_contain"]):
            continue

        filtered.append(r)

    return filtered


def has_active_filters(filters: dict) -> bool:
    return any(filters.values())
