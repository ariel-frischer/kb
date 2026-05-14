"""Terminal styling helpers for human-readable CLI output."""

import os
import sys

ANSI_RESET = "\033[0m"
ANSI_STYLES = {
    "answer": "36",
    "error": "31",
    "heading": "1",
    "label": "1;36",
    "metric": "33",
    "muted": "2",
    "path": "32",
    "rank": "35",
    "success": "32",
    "warning": "33",
}


def color_enabled() -> bool:
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    return sys.stdout.isatty()


def style(text: object, color: str) -> str:
    value = str(text)
    if not color_enabled():
        return value
    code = ANSI_STYLES[color]
    return f"\033[{code}m{value}{ANSI_RESET}"


def label(name: str, value: object) -> str:
    return f"{style(name + ':', 'label')} {value}"


def result_header(rank: int, path: str, meta: str) -> str:
    return (
        f"--- {style(f'[{rank}]', 'rank')} {style(path, 'path')} "
        f"{style(f'({meta})', 'muted')} ---"
    )


def print_error(message: str):
    print(style(message, "error"))
