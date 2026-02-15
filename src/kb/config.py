"""Configuration loading from .kb.toml and ~/.config/kb/secrets.yaml."""

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import yaml

CONFIG_FILE = ".kb.toml"
SECRETS_PATH = Path.home() / ".config" / "kb" / "secrets.yaml"
SCHEMA_VERSION = 3

CONFIG_TEMPLATE = """\
# Knowledge base config
# Run `kb init` to generate, `kb index` to index sources.

# Where to store the database (relative to this file)
db = "kb.db"

# Directories to index (relative to this file)
sources = [
    # "docs/",
    # "notes/",
]

# Embedding
# embed_model = "text-embedding-3-small"
# embed_dims = 1536

# LLM
# chat_model = "gpt-4o-mini"

# Chunking
# max_chunk_chars = 2000
# min_chunk_chars = 50

# Search
# min_similarity = 0.25   # cosine similarity floor for ask mode
# rrf_k = 60.0            # RRF smoothing constant
# rerank_fetch_k = 20     # candidates to fetch for LLM rerank
# rerank_top_k = 5        # how many to keep after rerank
"""


@dataclass
class Config:
    db: str = "kb.db"
    sources: list[str] = field(default_factory=list)
    embed_model: str = "text-embedding-3-small"
    embed_dims: int = 1536
    chat_model: str = "gpt-4o-mini"
    max_chunk_chars: int = 2000
    min_chunk_chars: int = 50
    min_similarity: float = 0.25
    rrf_k: float = 60.0
    rerank_fetch_k: int = 20
    rerank_top_k: int = 5

    # Resolved paths (set by find_config)
    config_dir: Path | None = None
    db_path: Path = field(default_factory=lambda: Path("kb.db"))

    @property
    def source_paths(self) -> list[Path]:
        if not self.config_dir:
            return []
        return [self.config_dir / s for s in self.sources]


def load_secrets() -> None:
    """Load API keys from ~/.config/kb/secrets.yaml into env vars.

    Existing env vars take precedence (never overwrite).
    Creates the config dir if it doesn't exist.
    """
    if not SECRETS_PATH.is_file():
        return
    with open(SECRETS_PATH) as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        return
    for key, value in data.items():
        env_key = key.upper()
        if env_key not in os.environ:
            os.environ[env_key] = str(value)


def find_config() -> Config:
    """Walk up from cwd to find .kb.toml. Returns Config with resolved paths."""
    cwd = Path.cwd()
    for parent in [cwd, *cwd.parents]:
        cfg_path = parent / CONFIG_FILE
        if cfg_path.is_file():
            with open(cfg_path, "rb") as f:
                data = tomllib.load(f)
            cfg = Config(**{k: v for k, v in data.items() if k in Config.__dataclass_fields__})
            cfg.config_dir = parent
            cfg.db_path = parent / cfg.db
            return cfg
        if parent == parent.parent:
            break
    # No config found — use defaults with cwd
    cfg = Config()
    cfg.db_path = cwd / cfg.db
    return cfg
