#!/usr/bin/env python3
"""Scaffold a new CHANGELOG.yaml entry from commits since the last tag.

Reads the current version from pyproject.toml, collects commits since the last
git tag, categorizes them by conventional commit prefix, and either prints a
YAML block to stdout or inserts it into CHANGELOG.yaml (with --write).
"""

from __future__ import annotations

import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

CHANGELOG_PATH = Path(__file__).resolve().parent.parent / "CHANGELOG.yaml"
PYPROJECT_PATH = Path(__file__).resolve().parent.parent / "pyproject.toml"

# Map conventional commit prefixes to changelog categories
PREFIX_MAP = {
    "feat": "added",
    "fix": "fixed",
    "refactor": "changed",
    "perf": "changed",
    "breaking": "breaking",
    "deprecate": "deprecated",
    "remove": "removed",
    "revert": "removed",
}


def run(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text=True).strip()


def get_version() -> str:
    """Read version from pyproject.toml."""
    text = PYPROJECT_PATH.read_text()
    m = re.search(r'^version\s*=\s*"(.+?)"', text, re.MULTILINE)
    if not m:
        print("error: could not parse version from pyproject.toml", file=sys.stderr)
        sys.exit(1)
    return m.group(1)


def get_last_tag() -> str | None:
    """Get the most recent git tag."""
    try:
        return run(["git", "describe", "--tags", "--abbrev=0"])
    except subprocess.CalledProcessError:
        return None


def get_commits_since(tag: str | None) -> list[tuple[str, str]]:
    """Return list of (short_sha, subject) since tag (or all if None)."""
    if tag:
        range_arg = f"{tag}..HEAD"
    else:
        range_arg = "HEAD"
    try:
        out = run(["git", "log", "--oneline", "--reverse", range_arg])
    except subprocess.CalledProcessError:
        return []
    if not out:
        return []
    result = []
    for line in out.splitlines():
        sha, _, subject = line.partition(" ")
        result.append((sha, subject))
    return result


def categorize(subject: str) -> str:
    """Map a commit subject to a changelog category."""
    # Match conventional commit prefix: type(scope): or type:
    m = re.match(r"(\w+)(?:\([^)]*\))?[!]?\s*:", subject)
    if m:
        prefix = m.group(1).lower()
        # Check for breaking change indicator
        if "!" in subject.split(":")[0]:
            return "breaking"
        return PREFIX_MAP.get(prefix, "changed")
    return "changed"


def clean_subject(subject: str) -> str:
    """Strip conventional commit prefix from subject for description."""
    cleaned = re.sub(r"^\w+(?:\([^)]*\))?[!]?\s*:\s*", "", subject)
    # Take only the first sentence (multi-line commit messages)
    first_line = cleaned.split("\n")[0].strip()
    # If there are multiple feat/fix in one commit message, take the first
    first_sentence = first_line.split(" fix(")[0].split(" feat(")[0].split(" chore(")[0]
    # Capitalize first letter
    if first_sentence:
        first_sentence = first_sentence[0].upper() + first_sentence[1:]
    return first_sentence


def build_entry(version: str, commits: list[tuple[str, str]]) -> str:
    """Build a YAML entry block."""
    categories: dict[str, list[dict]] = {
        "added": [],
        "fixed": [],
        "changed": [],
        "removed": [],
        "deprecated": [],
        "breaking": [],
    }

    for sha, subject in commits:
        # Skip chore/docs/style/test/ci commits
        m = re.match(r"(\w+)(?:\([^)]*\))?[!]?\s*:", subject)
        if m and m.group(1).lower() in ("chore", "docs", "style", "test", "ci"):
            continue
        cat = categorize(subject)
        desc = clean_subject(subject)
        if desc:
            categories[cat].append({"description": desc, "sha": sha})

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    lines = [
        f'  - version: "{version}"',
        f'    date: "{today}"',
    ]

    for cat in ("added", "fixed", "changed", "removed", "deprecated", "breaking"):
        items = categories[cat]
        if items:
            lines.append(f"    {cat}:")
            for item in items:
                lines.append(f'      - description: "{item["description"]}"')
                lines.append(f'        commits: ["{item["sha"]}"]')
        else:
            lines.append(f"    {cat}: []")

    lines.append("    migration: null")
    lines.append('    contributors: ["ariel-frischer"]')
    return "\n".join(lines)


def insert_into_changelog(entry: str) -> None:
    """Insert entry at the top of the releases list in CHANGELOG.yaml."""
    if not CHANGELOG_PATH.exists():
        CHANGELOG_PATH.write_text(f"releases:\n{entry}\n")
        return

    content = CHANGELOG_PATH.read_text()
    # Insert after "releases:" line
    idx = content.find("releases:\n")
    if idx == -1:
        print("error: could not find 'releases:' in CHANGELOG.yaml", file=sys.stderr)
        sys.exit(1)
    insert_pos = idx + len("releases:\n")
    new_content = content[:insert_pos] + entry + "\n\n" + content[insert_pos:]
    CHANGELOG_PATH.write_text(new_content)


def main() -> None:
    write_mode = "--write" in sys.argv
    version_arg = None
    for arg in sys.argv[1:]:
        if arg != "--write":
            version_arg = arg
            break

    version = version_arg or get_version()
    last_tag = get_last_tag()
    commits = get_commits_since(last_tag)

    if not commits:
        print(f"No commits found since {last_tag or 'beginning'}.", file=sys.stderr)
        sys.exit(0)

    print(f"Version: {version}", file=sys.stderr)
    print(f"Commits since {last_tag or 'beginning'}: {len(commits)}", file=sys.stderr)

    entry = build_entry(version, commits)

    if write_mode:
        insert_into_changelog(entry)
        print(f"Inserted entry for {version} into {CHANGELOG_PATH}", file=sys.stderr)
    else:
        print(entry)


if __name__ == "__main__":
    main()
