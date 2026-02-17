#!/usr/bin/env python3
"""Extract a version entry from CHANGELOG.yaml and render as markdown.

Usage: changelog_to_md.py <version>
       changelog_to_md.py 1.0.8

Prints markdown to stdout. If the entry has a `summary` field, it's printed
first as a paragraph, followed by a detailed breakdown.

Exits 0 even if version not found (empty output).
No external dependencies — parses the YAML subset used by CHANGELOG.yaml directly.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

CHANGELOG_PATH = Path(__file__).resolve().parent.parent / "CHANGELOG.yaml"

CATEGORY_TITLES = {
    "added": "Added",
    "fixed": "Fixed",
    "changed": "Changed",
    "removed": "Removed",
    "deprecated": "Deprecated",
    "breaking": "Breaking Changes",
}


def extract_version_block(text: str, version: str) -> str | None:
    """Extract the raw YAML block for a specific version."""
    # Match the version entry start: `  - version: "X.Y.Z"`
    pattern = rf'^  - version: "{re.escape(version)}"$'
    match = re.search(pattern, text, re.MULTILINE)
    if not match:
        return None

    start = match.start()
    # Find the next version entry or end of file
    next_entry = re.search(r"^  - version:", text[match.end() :], re.MULTILINE)
    if next_entry:
        end = match.end() + next_entry.start()
    else:
        end = len(text)

    return text[start:end].rstrip()


def parse_entry(block: str) -> dict:
    """Parse a version block into a dict with summary, categories, migration."""
    entry: dict = {"summary": None, "migration": None}

    # Extract summary (single-line string value)
    m = re.search(r'^\s+summary:\s*"(.+?)"$', block, re.MULTILINE)
    if m:
        entry["summary"] = m.group(1)

    # Extract migration
    m = re.search(r'^\s+migration:\s*"(.+?)"$', block, re.MULTILINE)
    if m:
        entry["migration"] = m.group(1)

    # Extract category items
    for cat in CATEGORY_TITLES:
        items = []
        # Find category section
        cat_match = re.search(rf"^\s+{cat}:\s*$", block, re.MULTILINE)
        if cat_match:
            # Collect description lines until next category or end
            rest = block[cat_match.end() :]
            for desc_match in re.finditer(r'^\s+- description:\s*"(.+?)"$', rest, re.MULTILINE):
                # Stop if we've hit the next top-level key
                preceding = rest[: desc_match.start()]
                if re.search(r"^\s+\w+:", preceding.split("\n")[-1] if preceding else "", re.MULTILINE):
                    # Check if it's a sub-key (commits:) or a new category
                    pass
                items.append(desc_match.group(1))
            # Stop collecting at next category
            filtered = []
            for line in rest.splitlines():
                # If line is a new top-level category, stop
                stripped = line.strip()
                if stripped and not stripped.startswith("-") and not stripped.startswith("commits:") and ":" in stripped:
                    key = stripped.split(":")[0]
                    if key in CATEGORY_TITLES or key in ("migration", "contributors"):
                        break
                m2 = re.match(r'\s+- description:\s*"(.+?)"', line)
                if m2:
                    filtered.append(m2.group(1))
            items = filtered
        entry[cat] = items

    return entry


def render_entry(entry: dict) -> str:
    """Render a parsed entry as markdown."""
    lines: list[str] = []

    if entry.get("summary"):
        lines.append(entry["summary"])
        lines.append("")

    for cat, title in CATEGORY_TITLES.items():
        items = entry.get(cat, [])
        if not items:
            continue
        lines.append(f"### {title}")
        for desc in items:
            lines.append(f"- {desc}")
        lines.append("")

    if entry.get("migration"):
        lines.append("### Migration")
        lines.append(entry["migration"])
        lines.append("")

    return "\n".join(lines).rstrip()


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: changelog_to_md.py <version>", file=sys.stderr)
        sys.exit(1)

    version = sys.argv[1].lstrip("v")

    if not CHANGELOG_PATH.exists():
        return

    text = CHANGELOG_PATH.read_text()
    block = extract_version_block(text, version)
    if not block:
        return

    entry = parse_entry(block)
    md = render_entry(entry)
    if md:
        print(md)


if __name__ == "__main__":
    main()
