"""Configuration loading from .kb.toml / ~/.config/kb/config.toml and secrets."""

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_CONFIG_FILE = ".kb.toml"
SECRETS_PATH = Path.home() / ".config" / "kb" / "secrets.toml"
SCHEMA_VERSION = 3

GLOBAL_CONFIG_DIR = Path.home() / ".config" / "kb"
GLOBAL_CONFIG_FILE = GLOBAL_CONFIG_DIR / "config.toml"
GLOBAL_DATA_DIR = Path.home() / ".local" / "share" / "kb"
GLOBAL_DB_PATH = GLOBAL_DATA_DIR / "kb.db"

PROJECT_CONFIG_TEMPLATE = """\
# Knowledge base config (project-local)
# Run `kb init --project` to generate, `kb index` to index sources.

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

GLOBAL_CONFIG_TEMPLATE = """\
# Knowledge base config (global)
# Manage sources with `kb add <dir>` / `kb remove <dir>`.

# Directories to index (absolute paths)
sources = [
    # "/home/user/notes",
    # "/home/user/docs",
]

# Embedding
# embed_model = "text-embedding-3-small"
# embed_dims = 1536

# LLM
# chat_model = "gpt-4o-mini"
"""

# Keep old name as alias for backward compat in imports
CONFIG_FILE = PROJECT_CONFIG_FILE
CONFIG_TEMPLATE = PROJECT_CONFIG_TEMPLATE


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

    scope: str = "project"  # "global" or "project"

    # Resolved paths (set by find_config)
    config_dir: Path | None = None
    config_path: Path | None = None
    db_path: Path = field(default_factory=lambda: Path("kb.db"))

    @property
    def source_paths(self) -> list[Path]:
        if self.scope == "global":
            return [Path(s).expanduser() for s in self.sources]
        if not self.config_dir:
            return []
        return [self.config_dir / s for s in self.sources]

    def doc_path_for_db(self, file_path: Path, source_dir: Path) -> str:
        """Compute the document path stored in the DB.

        Project mode: relative to config_dir (e.g. "docs/file.md").
        Global mode:  source_dir.name / relative (e.g. "notes/file.md").
        """
        if self.scope == "project" and self.config_dir:
            try:
                return str(file_path.relative_to(self.config_dir))
            except ValueError:
                pass
        # Global mode or fallback
        try:
            return str(Path(source_dir.name) / file_path.relative_to(source_dir))
        except ValueError:
            return str(file_path)


def load_secrets() -> None:
    """Load API keys from ~/.config/kb/secrets.toml into env vars.

    Existing env vars take precedence (never overwrite).
    """
    if not SECRETS_PATH.is_file():
        return
    with open(SECRETS_PATH, "rb") as f:
        data = tomllib.load(f)
    for key, value in data.items():
        env_key = key.upper()
        if env_key not in os.environ:
            os.environ[env_key] = str(value)


def _load_toml(cfg_path: Path, scope: str) -> Config:
    """Load a TOML config file and return a Config."""
    with open(cfg_path, "rb") as f:
        data = tomllib.load(f)
    cfg = Config(**{k: v for k, v in data.items() if k in Config.__dataclass_fields__})
    cfg.scope = scope
    cfg.config_dir = cfg_path.parent
    cfg.config_path = cfg_path
    if scope == "global":
        cfg.db_path = GLOBAL_DB_PATH
    else:
        cfg.db_path = cfg_path.parent / cfg.db
    return cfg


def find_config() -> Config:
    """Find config: project .kb.toml (walk-up) then global ~/.config/kb/config.toml."""
    # 1. Walk up from cwd for project config
    cwd = Path.cwd()
    for parent in [cwd, *cwd.parents]:
        cfg_path = parent / PROJECT_CONFIG_FILE
        if cfg_path.is_file():
            return _load_toml(cfg_path, "project")
        if parent == parent.parent:
            break

    # 2. Fall back to global config
    if GLOBAL_CONFIG_FILE.is_file():
        return _load_toml(GLOBAL_CONFIG_FILE, "global")

    # 3. No config found — return defaults
    cfg = Config()
    cfg.db_path = cwd / cfg.db
    return cfg


def _to_toml(data: dict) -> str:
    """Minimal TOML serializer for our flat config."""
    lines = []
    for key, val in data.items():
        if isinstance(val, str):
            lines.append(f'{key} = "{val}"')
        elif isinstance(val, bool):
            lines.append(f"{key} = {'true' if val else 'false'}")
        elif isinstance(val, (int, float)):
            lines.append(f"{key} = {val}")
        elif isinstance(val, list):
            items = ", ".join(f'"{v}"' for v in val)
            lines.append(f"{key} = [{items}]")
    return "\n".join(lines) + "\n"


def save_config(cfg: Config) -> None:
    """Save non-default config values to the config file."""
    if not cfg.config_path:
        return
    defaults = Config()
    data: dict = {}
    # Only write fields that differ from defaults (skip internal fields)
    skip = {"scope", "config_dir", "config_path", "db_path"}
    for fname, fld in Config.__dataclass_fields__.items():
        if fname in skip:
            continue
        val = getattr(cfg, fname)
        default_val = getattr(defaults, fname)
        if val != default_val:
            data[fname] = val
    cfg.config_path.write_text(_to_toml(data))
