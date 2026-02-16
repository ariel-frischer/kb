"""Shared fixtures for kb tests."""

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import sqlite_vec

from kb.config import Config


@pytest.fixture
def tmp_config(tmp_path):
    """Config pointing at a temp directory with project scope."""
    cfg = Config()
    cfg.scope = "project"
    cfg.config_dir = tmp_path
    cfg.config_path = tmp_path / ".kb.toml"
    cfg.db_path = tmp_path / "kb.db"
    return cfg


@pytest.fixture
def global_config(tmp_path):
    """Config with global scope."""
    cfg = Config()
    cfg.scope = "global"
    cfg.config_dir = tmp_path
    cfg.config_path = tmp_path / "config.toml"
    cfg.db_path = tmp_path / "kb.db"
    return cfg


@pytest.fixture
def db_conn(tmp_config):
    """Live sqlite connection with schema created (via db.connect)."""
    from kb.db import connect

    conn = connect(tmp_config)
    yield conn
    conn.close()


@pytest.fixture
def mock_openai():
    """Mock OpenAI client."""
    client = MagicMock()
    return client


@pytest.fixture
def sample_markdown():
    return """\
# Project Overview

This is a sample project with multiple sections.

## Installation

Run pip install to get started. The package supports Python 3.12+.

## Usage

### Basic Commands

Use the CLI to search your knowledge base:

```
kb search "my query"
kb ask "my question"
```

### Advanced Features

Filters let you narrow results by file, date, or keywords.

## Contributing

Open a PR on GitHub. Follow the code style guide.
"""


@pytest.fixture
def sample_chunks():
    """Pre-built search result dicts for testing."""
    return [
        {
            "chunk_id": 1,
            "rrf_score": 0.05,
            "distance": 0.3,
            "similarity": 0.7,
            "text": "Install with pip install kb",
            "doc_path": "docs/install.md",
            "heading": "Installation",
            "in_fts": True,
            "in_vec": True,
        },
        {
            "chunk_id": 2,
            "rrf_score": 0.03,
            "distance": 0.5,
            "similarity": 0.5,
            "text": "Search your knowledge base with kb search",
            "doc_path": "docs/usage.md",
            "heading": "Search",
            "in_fts": False,
            "in_vec": True,
        },
        {
            "chunk_id": 3,
            "rrf_score": 0.02,
            "distance": 0.6,
            "similarity": 0.4,
            "text": "Contributing guidelines for the project",
            "doc_path": "articles/contrib.md",
            "heading": "Contributing",
            "in_fts": True,
            "in_vec": True,
        },
    ]
