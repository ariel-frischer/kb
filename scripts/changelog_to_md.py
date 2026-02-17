#!/usr/bin/env python3
"""Extract a version entry from CHANGELOG.yaml and render as markdown.

Usage: changelog_to_md.py <version>
       changelog_to_md.py 1.0.8

Prints markdown to stdout. If the entry has a `summary` field, it's printed
first as a paragraph, followed by a detailed breakdown. If no summary exists,
just the detailed breakdown is printed.

Exits 0 even if version not found (empty output).
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

CHANGELOG_PATH = Path(__file__).resolve().parent.parent / "CHANGELOG.yaml"

CATEGORY_TITLES = {
    "added": "Added",
    "fixed": "Fixed",
    "changed": "Changed",
    "removed": "Removed",
    "deprecated": "Deprecated",
    "breaking": "Breaking Changes",
}


def render_entry(entry: dict) -> str:
    """Render a single release entry as markdown."""
    lines: list[str] = []

    summary = entry.get("summary")
    if summary:
        lines.append(summary)
        lines.append("")

    has_details = False
    for key, title in CATEGORY_TITLES.items():
        items = entry.get(key, [])
        if not items:
            continue
        has_details = True
        lines.append(f"### {title}")
        for item in items:
            desc = item.get("description", "") if isinstance(item, dict) else str(item)
            lines.append(f"- {desc}")
        lines.append("")

    migration = entry.get("migration")
    if migration:
        lines.append("### Migration")
        lines.append(migration)
        lines.append("")

    if summary and not has_details:
        return summary.rstrip()

    return "\n".join(lines).rstrip()


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: changelog_to_md.py <version>", file=sys.stderr)
        sys.exit(1)

    version = sys.argv[1].lstrip("v")

    if not CHANGELOG_PATH.exists():
        return

    data = yaml.safe_load(CHANGELOG_PATH.read_text())
    releases = data.get("releases", [])

    for entry in releases:
        if entry.get("version") == version:
            md = render_entry(entry)
            if md:
                print(md)
            return


if __name__ == "__main__":
    main()
