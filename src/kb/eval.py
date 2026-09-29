"""BEIR retrieval benchmarks for `kb eval`, with a cumulative API spend cap.

Scores kb's real retrieval pipeline (FTS, hybrid search, reranking) against public
BEIR datasets using their shipped relevance judgments (qrels). Datasets, corpus
files, isolated indexes, run reports, and the spend ledger all live under
``GLOBAL_DATA_DIR/eval/`` — the user's configured database is never touched.
"""

from __future__ import annotations

import contextlib
import gzip
import hashlib
import io
import json
import math
import re
import shutil
import statistics
import time
import urllib.error
import urllib.request
import uuid
import zipfile
from collections.abc import Callable, Iterable
from copy import copy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI

from .api import KBError, NoSearchTermsError, fts_core, search_core
from .chunk import CHONKIE_AVAILABLE, chunk_markdown, embedding_text
from .config import GLOBAL_DATA_DIR, Config
from .cost import chat_cost_usd, embed_cost_usd, estimate_tokens
from .db import connect
from .expand import _LLM_PROMPT as EXPAND_PROMPT
from .expand import EXPAND_MAX_TOKENS
from .extract import extract_text
from .hyde import _SYSTEM_PROMPT as HYDE_SYSTEM_PROMPT
from .hyde import HYDE_MAX_TOKENS
from .ingest import index_directory, md5_hash
from .llm import CHATGPT, hyde_provider, load_chatgpt_credentials
from .rerank import (
    RERANK_MAX_TOKENS,
    RERANK_PASSAGE_CHARS,
    RERANK_SYSTEM_PROMPT,
    rerank,
)

EVAL_DIR = GLOBAL_DATA_DIR / "eval"

K = 10
MODES = ("fts", "hybrid", "rerank")
DEFAULT_MODES = ("fts", "hybrid")
VECTOR_MODES = frozenset({"hybrid", "rerank"})
# Chunks fetched per query before collapsing to documents (docs can span chunks).
FETCH_CHUNKS = 3 * K
# Multiplier on chars/4 token estimates so per-query cost bounds stay conservative.
TOKEN_SAFETY = 1.5
# Upper bound on the per-passage label ("[n] (path > heading)") in rerank prompts.
RERANK_LABEL_CHARS = 200

UKP_URL = (
    "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"
)
HF_URL = "https://huggingface.co/datasets/BeIR/{repo}/resolve/{revision}/{file}"

# Pinned Hugging Face mirror revisions (last commits that still ship the JSONL files;
# the mirrors were later converted to parquet). Values: (corpus/queries repo, qrels repo).
DATASETS: dict[str, tuple[str, str]] = {
    "scifact": (
        "984eed826375f18d27936c4a32bf0f8491e3f414",
        "2938d17dc3b09882fdb8c12bbbe2e2dc0e75a029",
    ),
    "nfcorpus": (
        "e28763e68d85db0fa71652ba3c4afabf8c5b3bb7",
        "a451b3b26d3ae1358f259c1a3a4dd61fcea35a65",
    ),
    "arguana": (
        "900330faa8ca2828a468dfe795f9d3c3887c8cfc",
        "ae5468c6f1c198109a8af5f0d4dc58bd18b6fea7",
    ),
    "fiqa": (
        "e31855201132ea2a257d7df77c828d7c02427521",
        "252958f2d646e22cab6d0c72dd3f0d5de6d0655a",
    ),
    "scidocs": (
        "7abe49caf46c871ac644ffa3d3ba362d01290afb",
        "735ea1048e37b1ebce14c6dc3d33a5edaf66d3dc",
    ),
}
DEFAULT_DATASET = "scifact"

_RAW_FILES = ("corpus.jsonl", "queries.jsonl", "qrels/test.tsv")

Progress = Callable[[str], None]


class EvalError(KBError):
    """Raised when an eval run cannot start or complete."""


class EvalBudgetError(EvalError):
    """Raised when an eval run would exceed (or cannot enforce) the spend cap."""


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def eval_dir() -> Path:
    return EVAL_DIR


def dataset_dir(name: str) -> Path:
    return eval_dir() / "datasets" / name


def ledger_path() -> Path:
    return eval_dir() / "spend.json"


# ---------------------------------------------------------------------------
# Dataset download + loading
# ---------------------------------------------------------------------------


@dataclass
class BeirData:
    corpus: dict[str, tuple[str, str]]  # doc_id -> (title, text)
    queries: dict[str, str]  # query_id -> text (only queries with test qrels)
    qrels: dict[str, dict[str, int]]  # query_id -> {doc_id: relevance > 0}


def _download(url: str, dest: Path, timeout: float = 120) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "kb-eval"})
    with urllib.request.urlopen(req, timeout=timeout) as resp, open(tmp, "wb") as f:
        shutil.copyfileobj(resp, f)
    tmp.replace(dest)


def extract_beir_zip(zip_path: Path, raw_dir: Path) -> None:
    """Extract corpus.jsonl, queries.jsonl, qrels/test.tsv from a BEIR zip into raw_dir."""
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        for target in _RAW_FILES:
            matches = sorted(
                (n for n in names if n == target or n.endswith("/" + target)),
                key=len,
            )
            if not matches:
                raise EvalError(f"{zip_path.name} is missing {target}")
            dest = raw_dir / target
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_name(dest.name + ".part")
            with zf.open(matches[0]) as src, open(tmp, "wb") as out:
                shutil.copyfileobj(src, out)
            tmp.replace(dest)


def _download_ukp(name: str, raw_dir: Path) -> None:
    zip_path = raw_dir.parent / f"{name}.zip"
    try:
        _download(UKP_URL.format(name=name), zip_path)
        extract_beir_zip(zip_path, raw_dir)
    finally:
        zip_path.unlink(missing_ok=True)


def _download_hf(name: str, raw_dir: Path) -> None:
    revision, qrels_revision = DATASETS[name]
    for stem in ("corpus", "queries"):
        gz_path = raw_dir / f"{stem}.jsonl.gz"
        try:
            _download(
                HF_URL.format(repo=name, revision=revision, file=f"{stem}.jsonl.gz"),
                gz_path,
            )
            dest = raw_dir / f"{stem}.jsonl"
            tmp = dest.with_name(dest.name + ".part")
            with gzip.open(gz_path, "rb") as src, open(tmp, "wb") as out:
                shutil.copyfileobj(src, out)
            tmp.replace(dest)
        finally:
            gz_path.unlink(missing_ok=True)
    _download(
        HF_URL.format(repo=f"{name}-qrels", revision=qrels_revision, file="test.tsv"),
        raw_dir / "qrels" / "test.tsv",
    )


def ensure_dataset(name: str, progress: Progress | None = None) -> Path:
    """Download (once) and return the raw BEIR directory for a dataset."""
    raw_dir = dataset_dir(name) / "raw"
    if all((raw_dir / f).is_file() for f in _RAW_FILES):
        return raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)
    _notify(progress, f"Downloading BEIR {name} from {UKP_URL.format(name=name)}")
    try:
        _download_ukp(name, raw_dir)
        return raw_dir
    except (OSError, urllib.error.URLError, zipfile.BadZipFile, EvalError) as e:
        _notify(progress, f"UKP download failed ({e}); trying Hugging Face mirror")
    try:
        _download_hf(name, raw_dir)
    except (OSError, urllib.error.URLError) as e:
        raise EvalError(f"Could not download BEIR dataset '{name}': {e}") from e
    return raw_dir


def _read_jsonl(path: Path) -> Iterable[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_qrels(path: Path) -> dict[str, dict[str, int]]:
    """Parse a BEIR qrels TSV, keeping only positive judgments."""
    qrels: dict[str, dict[str, int]] = {}
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            qid, doc_id, score = parts[0], parts[1], parts[2]
            if i == 0 and not score.lstrip("-").isdigit():
                continue  # header row
            rel = int(score)
            if rel > 0:
                qrels.setdefault(qid, {})[doc_id] = rel
    return qrels


def load_beir(raw_dir: Path) -> BeirData:
    """Load a BEIR dataset directory (corpus.jsonl, queries.jsonl, qrels/test.tsv)."""
    qrels = load_qrels(raw_dir / "qrels" / "test.tsv")
    corpus = {
        str(d["_id"]): ((d.get("title") or "").strip(), d.get("text") or "")
        for d in _read_jsonl(raw_dir / "corpus.jsonl")
    }
    queries = {
        str(q["_id"]): q.get("text") or ""
        for q in _read_jsonl(raw_dir / "queries.jsonl")
        if str(q["_id"]) in qrels
    }
    return BeirData(corpus=corpus, queries=queries, qrels=qrels)


# ---------------------------------------------------------------------------
# Corpus materialization + isolated index
# ---------------------------------------------------------------------------


def safe_doc_id(doc_id: str) -> str:
    """Filesystem-safe, collision-free file stem for a BEIR doc id."""
    safe = re.sub(r"[^A-Za-z0-9_-]", "_", doc_id)
    if safe == doc_id and 0 < len(safe) <= 100:
        return safe
    return f"{safe[:80]}-{hashlib.md5(doc_id.encode()).hexdigest()[:8]}"


def materialize_corpus(name: str, corpus: dict[str, tuple[str, str]]) -> Path:
    """Write one markdown file per doc (``# title`` + text). Returns the corpus dir."""
    base = dataset_dir(name)
    corpus_dir = base / "corpus"
    marker = base / "corpus.json"
    if marker.is_file() and json.loads(marker.read_text()).get("docs") == len(corpus):
        return corpus_dir
    corpus_dir.mkdir(parents=True, exist_ok=True)
    for doc_id, (title, text) in corpus.items():
        title = " ".join(title.split())
        body = f"# {title}\n\n{text}\n" if title else f"{text}\n"
        (corpus_dir / f"{safe_doc_id(doc_id)}.md").write_text(body, encoding="utf-8")
    marker.write_text(json.dumps({"docs": len(corpus)}))
    return corpus_dir


def index_fingerprint(cfg: Config) -> str:
    """Short hash of settings that change the index contents."""
    key = {
        "embed_method": cfg.embed_method,
        "embed_model": cfg.local_embed_model
        if cfg.embed_method == "local"
        else cfg.embed_model,
        "embed_dims": cfg.embed_dims,
        "max_chunk_chars": cfg.max_chunk_chars,
        "min_chunk_chars": cfg.min_chunk_chars,
        "chonkie": CHONKIE_AVAILABLE,
    }
    return hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:12]


def eval_config(cfg: Config, name: str, corpus_dir: Path) -> Config:
    """Copy of the user's config pointed at an isolated eval index for ``name``."""
    ecfg = copy(cfg)
    ecfg.scope = "project"
    ecfg.sources = ["."]
    ecfg.allowed_large_files = []
    ecfg.index_code = False
    ecfg.config_dir = corpus_dir
    ecfg.config_path = None
    ecfg.db_path = eval_dir() / "indexes" / f"{name}-{index_fingerprint(cfg)}" / "kb.db"
    return ecfg


def _indexed_hashes(ecfg: Config) -> dict[str, str]:
    if not ecfg.db_path.exists():
        return {}
    conn = connect(ecfg)
    try:
        return {
            r["path"]: r["content_hash"]
            for r in conn.execute("SELECT path, content_hash FROM documents")
        }
    finally:
        conn.close()


def pending_embedding_texts(ecfg: Config, corpus_dir: Path) -> list[str]:
    """Embedding inputs the next index run would send (mirrors ingest._index_file)."""
    indexed = _indexed_hashes(ecfg)
    texts: list[str] = []
    for path in sorted(corpus_dir.glob("*.md")):
        rel_path = ecfg.doc_path_for_db(path, corpus_dir)
        extracted = extract_text(path)
        if extracted is None:
            continue
        text, _ = extracted
        if len(text.strip()) < ecfg.min_chunk_chars:
            continue
        if indexed.get(rel_path) == md5_hash(text):
            continue
        for chunk in chunk_markdown(text, ecfg):
            ancestry = chunk.get("heading_ancestry", "")
            texts.append(embedding_text(chunk["text"], ancestry, rel_path))
    return texts


def _max_chunk_id(ecfg: Config) -> int:
    if not ecfg.db_path.exists():
        return 0
    conn = connect(ecfg)
    try:
        return conn.execute("SELECT COALESCE(MAX(id), 0) FROM chunks").fetchone()[0]
    finally:
        conn.close()


def _embedded_tokens_since(ecfg: Config, chunk_id: int) -> int:
    conn = connect(ecfg)
    try:
        rows = conn.execute(
            "SELECT text, heading_ancestry, doc_path FROM chunks WHERE id > ?",
            (chunk_id,),
        ).fetchall()
    finally:
        conn.close()
    return sum(
        estimate_tokens(
            embedding_text(r["text"], r["heading_ancestry"] or "", r["doc_path"])
        )
        for r in rows
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def collapse_to_docs(
    doc_paths: Iterable[str | None],
    id_map: dict[str, str],
    *,
    exclude: str | None = None,
    k: int = K,
) -> list[str]:
    """Map ranked chunk doc_paths to BEIR doc ids, keeping each doc's first rank.

    ``exclude`` drops a doc id (BEIR ignores a query's own id, e.g. ArguAna).
    """
    ranked: list[str] = []
    seen: set[str] = set()
    for path in doc_paths:
        if not path:
            continue
        stem = Path(path).stem
        doc_id = id_map.get(stem, stem)
        if doc_id in seen or doc_id == exclude:
            continue
        seen.add(doc_id)
        ranked.append(doc_id)
        if len(ranked) == k:
            break
    return ranked


def ndcg_at_k(ranked: list[str], rels: dict[str, int], k: int = K) -> float:
    """nDCG@k with graded (linear) gains, as in trec_eval / BEIR."""
    dcg = sum(
        rels.get(doc_id, 0) / math.log2(i + 2) for i, doc_id in enumerate(ranked[:k])
    )
    ideal = sorted(rels.values(), reverse=True)[:k]
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def recall_at_k(ranked: list[str], rels: dict[str, int], k: int = K) -> float:
    if not rels:
        return 0.0
    return sum(1 for d in ranked[:k] if rels.get(d, 0) > 0) / len(rels)


def mrr_at_k(ranked: list[str], rels: dict[str, int], k: int = K) -> float:
    for i, doc_id in enumerate(ranked[:k]):
        if rels.get(doc_id, 0) > 0:
            return 1.0 / (i + 1)
    return 0.0


def precision_at_k(ranked: list[str], rels: dict[str, int], k: int = K) -> float:
    return sum(1 for d in ranked[:k] if rels.get(d, 0) > 0) / k


def score_query(ranked: list[str], rels: dict[str, int], k: int = K) -> dict:
    return {
        f"ndcg@{k}": ndcg_at_k(ranked, rels, k),
        f"recall@{k}": recall_at_k(ranked, rels, k),
        f"mrr@{k}": mrr_at_k(ranked, rels, k),
        f"p@{k}": precision_at_k(ranked, rels, k),
    }


def summarize(scores: list[dict], latencies_ms: list[float]) -> dict:
    out: dict = {"queries": len(scores)}
    for key in (f"ndcg@{K}", f"recall@{K}", f"mrr@{K}", f"p@{K}"):
        out[key] = (
            round(sum(s[key] for s in scores) / len(scores), 4) if scores else 0.0
        )
    out["latency_p50_ms"] = (
        round(statistics.median(latencies_ms)) if latencies_ms else 0
    )
    return out


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------


class SpendLedger:
    """Cumulative eval API spend, persisted across runs at ``eval/spend.json``."""

    def __init__(self, path: Path):
        self.path = path

    def _load(self) -> dict:
        if not self.path.is_file():
            return {"total_usd": 0.0, "runs": {}}
        return json.loads(self.path.read_text())

    def total(self) -> float:
        return float(self._load().get("total_usd", 0.0))

    def add(self, usd: float, *, run_id: str, dataset: str) -> float:
        """Record spend for a run and return the new cumulative total."""
        if usd <= 0:
            return self.total()
        data = self._load()
        data["total_usd"] = float(data.get("total_usd", 0.0)) + usd
        run = data.setdefault("runs", {}).setdefault(
            run_id, {"dataset": dataset, "usd": 0.0}
        )
        run["usd"] += usd
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".part")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(self.path)
        return data["total_usd"]


@dataclass(frozen=True)
class ApiUsage:
    """Which pipeline steps call a paid API for the selected modes."""

    embed: bool
    hyde: bool
    expand: bool
    rerank: bool
    # LLM steps billed to the ChatGPT subscription ($0, but need Codex credentials).
    subscription: bool = False

    @property
    def any(self) -> bool:
        return self.embed or self.hyde or self.expand or self.rerank


def api_usage(cfg: Config, modes: Iterable[str]) -> ApiUsage:
    vector = bool(VECTOR_MODES & set(modes))
    hyde_llm = vector and cfg.hyde_enabled and cfg.hyde_method != "local"
    expand_llm = vector and cfg.query_expand and cfg.expand_method == "llm"
    rerank_llm = "rerank" in modes and cfg.rerank_method != "cross-encoder"
    hyde_paid = hyde_provider(cfg) != CHATGPT
    chat_paid = cfg.llm_provider != CHATGPT
    return ApiUsage(
        # Indexing always embeds the corpus, even for fts-only runs.
        embed=cfg.embed_method != "local",
        hyde=hyde_llm and hyde_paid,
        expand=expand_llm and chat_paid,
        rerank=rerank_llm and chat_paid,
        subscription=(hyde_llm and not hyde_paid)
        or ((expand_llm or rerank_llm) and not chat_paid),
    )


def unpriced_models(cfg: Config, usage: ApiUsage) -> list[str]:
    """API models used by this run that have no known price (cap unenforceable)."""
    missing = []
    if usage.embed and embed_cost_usd(cfg.embed_model, 1) is None:
        missing.append(f"embed_model={cfg.embed_model}")
    hyde_model = cfg.hyde_model or cfg.chat_model
    if usage.hyde and chat_cost_usd(hyde_model, 1, 1) is None:
        missing.append(f"hyde model={hyde_model}")
    if (usage.expand or usage.rerank) and chat_cost_usd(cfg.chat_model, 1, 1) is None:
        missing.append(f"chat_model={cfg.chat_model}")
    return missing


def _padded(tokens: int) -> int:
    return math.ceil(tokens * TOKEN_SAFETY)


def estimate_query_cost_usd(query: str, cfg: Config, usage: ApiUsage, modes) -> float:
    """Conservative upper bound on API spend for one query across vector modes."""
    if not VECTOR_MODES & set(modes):
        return 0.0
    usd = 0.0
    if usage.hyde:
        prompt = _padded(estimate_tokens(HYDE_SYSTEM_PROMPT) + estimate_tokens(query))
        usd += chat_cost_usd(cfg.hyde_model or cfg.chat_model, prompt, HYDE_MAX_TOKENS)
    if usage.expand:
        prompt = _padded(estimate_tokens(EXPAND_PROMPT.format(query=query)))
        usd += chat_cost_usd(cfg.chat_model, prompt, EXPAND_MAX_TOKENS)
    if usage.embed:
        tokens = estimate_tokens(query)
        if cfg.hyde_enabled:
            tokens += HYDE_MAX_TOKENS
        if cfg.query_expand:
            tokens += EXPAND_MAX_TOKENS
        usd += embed_cost_usd(cfg.embed_model, _padded(tokens))
    if usage.rerank:
        passages_chars = cfg.rerank_fetch_k * (
            RERANK_PASSAGE_CHARS + RERANK_LABEL_CHARS
        )
        prompt = _padded(
            estimate_tokens(RERANK_SYSTEM_PROMPT)
            + estimate_tokens(query)
            + math.ceil(passages_chars / 4)
        )
        usd += chat_cost_usd(cfg.chat_model, prompt, RERANK_MAX_TOKENS)
    return usd


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _notify(progress: Progress | None, message: str) -> None:
    if progress:
        progress(message)


def parse_modes(modes: Iterable[str] | str | None) -> list[str]:
    if modes is None:
        return list(DEFAULT_MODES)
    if isinstance(modes, str):
        modes = modes.split(",")
    selected = [m.strip() for m in modes if m.strip()]
    unknown = [m for m in selected if m not in MODES]
    if unknown or not selected:
        raise EvalError(
            f"Unknown mode(s): {', '.join(unknown) or '(none)'}. "
            f"Choose from: {', '.join(MODES)}"
        )
    return list(dict.fromkeys(selected))


def _add_llm_tokens(totals: dict[str, int], items: Iterable[dict]) -> None:
    for item in items:
        if "prompt_tokens" in item:
            totals["prompt"] += item["prompt_tokens"]
            totals["completion"] += item["completion_tokens"]


def _rerank_all(
    client: OpenAI | None, question: str, candidates: list[dict], cfg: Config
) -> tuple[list[dict], dict]:
    """Rerank every candidate (rerank() keeps only rerank_top_k, so re-append the rest)."""
    if len(candidates) < 2:
        return candidates, {}
    rcfg = copy(cfg)
    # rerank() is a no-op when len(results) <= rerank_top_k.
    rcfg.rerank_top_k = len(candidates) - 1
    head, info = rerank(client, question, candidates, rcfg)
    kept = {id(r) for r in head}
    return head + [r for r in candidates if id(r) not in kept], info


def _config_summary(cfg: Config, modes: list[str]) -> dict:
    return {
        "embed_method": cfg.embed_method,
        "embed_model": cfg.local_embed_model
        if cfg.embed_method == "local"
        else cfg.embed_model,
        "hyde": (cfg.hyde_method if cfg.hyde_enabled else "off")
        if VECTOR_MODES & set(modes)
        else "n/a",
        "query_expand": (cfg.expand_method if cfg.query_expand else "off")
        if VECTOR_MODES & set(modes)
        else "n/a",
        "rerank_method": cfg.rerank_method if "rerank" in modes else "n/a",
        "chat_model": cfg.chat_model,
        "llm_provider": cfg.llm_provider,
    }


def run_eval(
    dataset: str,
    cfg: Config,
    *,
    modes: Iterable[str] | str | None = None,
    limit: int | None = None,
    budget: float | None = None,
    progress: Progress | None = None,
) -> dict:
    """Benchmark kb retrieval on a BEIR dataset. Returns a report dict."""
    if dataset not in DATASETS:
        raise EvalError(
            f"Unknown dataset '{dataset}'. Choose from: {', '.join(DATASETS)}"
        )
    mode_list = parse_modes(modes)
    budget_usd = cfg.eval_budget_usd if budget is None else budget
    if budget_usd < 0:
        raise EvalError("Budget must be >= 0.")
    if limit is not None and limit < 1:
        raise EvalError("--limit must be >= 1.")

    usage = api_usage(cfg, mode_list)
    if usage.subscription:
        # Fail before any download/indexing if the Codex login is missing/expired.
        load_chatgpt_credentials()
    unpriced = unpriced_models(cfg, usage)
    if unpriced:
        raise EvalBudgetError(
            "Refusing to run: no known price for "
            + ", ".join(unpriced)
            + ", so the eval spend cap cannot be enforced. Use a priced model "
            "(see kb/cost.py) or local methods (embed_method/hyde_method/"
            'expand_method = "local", rerank_method = "cross-encoder").'
        )

    ledger = SpendLedger(ledger_path())
    run_started = datetime.now(timezone.utc)
    run_id = (
        f"{run_started.strftime('%Y%m%dT%H%M%SZ')}-{dataset}-{uuid.uuid4().hex[:6]}"
    )

    raw_dir = ensure_dataset(dataset, progress)
    data = load_beir(raw_dir)
    corpus_dir = materialize_corpus(dataset, data.corpus)
    id_map = {safe_doc_id(doc_id): doc_id for doc_id in data.corpus}
    ecfg = eval_config(cfg, dataset, corpus_dir)

    query_ids = list(data.queries)
    if limit is not None:
        query_ids = query_ids[:limit]

    # --- Preflight -------------------------------------------------------
    corpus_estimate = 0.0
    if usage.embed:
        pending = pending_embedding_texts(ecfg, corpus_dir)
        corpus_estimate = embed_cost_usd(
            cfg.embed_model, sum(estimate_tokens(t) for t in pending)
        )
    query_estimates = {
        qid: estimate_query_cost_usd(data.queries[qid], ecfg, usage, mode_list)
        for qid in query_ids
    }
    preflight = corpus_estimate + sum(query_estimates.values())
    spent_before = ledger.total()
    remaining = budget_usd - spent_before
    if preflight > 0 and preflight > remaining:
        raise EvalBudgetError(
            f"Refusing to run: estimated API cost ${preflight:.4f} exceeds the "
            f"remaining eval budget ${max(remaining, 0.0):.4f} "
            f"(budget ${budget_usd:.2f}, already spent ${spent_before:.4f}). "
            "Raise --budget / eval_budget_usd, use --limit, or use local methods."
        )

    # --- Index -----------------------------------------------------------
    run_usd = 0.0
    index_usd = 0.0
    _notify(
        progress,
        f"Indexing {len(data.corpus)} {dataset} docs into {ecfg.db_path} "
        "(unchanged docs are skipped)",
    )
    before_id = _max_chunk_id(ecfg)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            index_directory(corpus_dir, ecfg)
    except BaseException:
        # Batches may have been paid for before the failure; charge the estimate.
        ledger.add(corpus_estimate, run_id=run_id, dataset=dataset)
        raise
    if usage.embed:
        index_usd = embed_cost_usd(
            cfg.embed_model, _embedded_tokens_since(ecfg, before_id)
        )
        ledger.add(index_usd, run_id=run_id, dataset=dataset)
        run_usd += index_usd

    # --- Queries ---------------------------------------------------------
    vector = bool(VECTOR_MODES & set(mode_list))
    fetch_k = (
        max(FETCH_CHUNKS, ecfg.rerank_fetch_k)
        if "rerank" in mode_list
        else FETCH_CHUNKS
    )
    search_bucket = "hybrid" if "hybrid" in mode_list else "rerank"
    rerank_client = OpenAI() if usage.rerank else None
    scores: dict[str, list[dict]] = {m: [] for m in mode_list}
    latencies: dict[str, list[float]] = {m: [] for m in mode_list}
    mode_usd: dict[str, float] = {m: 0.0 for m in mode_list}
    budget_exhausted = False
    largest_query_usd = 0.0
    llm_tokens = {"prompt": 0, "completion": 0}

    _notify(progress, f"Evaluating {len(query_ids)} queries: {', '.join(mode_list)}")
    for n, qid in enumerate(query_ids, 1):
        bound = max(query_estimates[qid], largest_query_usd)
        if bound > 0 and ledger.total() + bound > budget_usd:
            budget_exhausted = True
            _notify(progress, f"Budget reached after {n - 1} queries; stopping.")
            break

        query_usd = 0.0
        try:
            text = data.queries[qid]
            rels = data.qrels[qid]

            if "fts" in mode_list:
                t0 = time.perf_counter()
                try:
                    fts_results = fts_core(text, ecfg, top_k=FETCH_CHUNKS)["results"]
                except NoSearchTermsError:
                    fts_results = []
                latencies["fts"].append((time.perf_counter() - t0) * 1000)
                ranked = collapse_to_docs(
                    (r["doc_path"] for r in fts_results), id_map, exclude=qid
                )
                scores["fts"].append(score_query(ranked, rels))

            if vector:
                t0 = time.perf_counter()
                result = search_core(text, ecfg, top_k=fetch_k)
                search_ms = (time.perf_counter() - t0) * 1000
                search_cost = result.get("cost", {})
                search_usd = search_cost.get("estimated_total_usd", 0.0)
                mode_usd[search_bucket] += search_usd
                query_usd += search_usd
                _add_llm_tokens(llm_tokens, search_cost.get("items", []))
                candidates = result["results"]
                if "hybrid" in mode_list:
                    latencies["hybrid"].append(search_ms)
                    ranked = collapse_to_docs(
                        (r["doc_path"] for r in candidates), id_map, exclude=qid
                    )
                    scores["hybrid"].append(score_query(ranked, rels))
                if "rerank" in mode_list:
                    t0 = time.perf_counter()
                    reranked, info = _rerank_all(
                        rerank_client, text, candidates[: ecfg.rerank_fetch_k], ecfg
                    )
                    rerank_ms = (time.perf_counter() - t0) * 1000
                    if "prompt_tokens" in info:
                        _add_llm_tokens(llm_tokens, [info])
                    if usage.rerank and "prompt_tokens" in info:
                        rerank_usd = chat_cost_usd(
                            ecfg.chat_model,
                            info["prompt_tokens"],
                            info["completion_tokens"],
                        )
                        mode_usd["rerank"] += rerank_usd
                        query_usd += rerank_usd
                    latencies["rerank"].append(search_ms + rerank_ms)
                    ranked = collapse_to_docs(
                        (r["doc_path"] for r in reranked), id_map, exclude=qid
                    )
                    scores["rerank"].append(score_query(ranked, rels))
        except BaseException:
            # Calls may have been billed before the failure; charge the upper bound.
            ledger.add(max(bound, query_usd), run_id=run_id, dataset=dataset)
            raise

        if query_usd > 0:
            ledger.add(query_usd, run_id=run_id, dataset=dataset)
            run_usd += query_usd
            largest_query_usd = max(largest_query_usd, query_usd)
        if n % 50 == 0:
            _notify(progress, f"  {n}/{len(query_ids)} queries")

    ledger_total = ledger.total()
    report = {
        "run_id": run_id,
        "dataset": dataset,
        "split": "test",
        "k": K,
        "modes": mode_list,
        "corpus_docs": len(data.corpus),
        "queries_total": len(data.queries),
        "queries_selected": len(query_ids),
        "budget_exhausted": budget_exhausted,
        "metrics": {
            m: {**summarize(scores[m], latencies[m]), "usd": round(mode_usd[m], 6)}
            for m in mode_list
        },
        "spend": {
            "run_usd": round(run_usd, 6),
            "index_usd": round(index_usd, 6),
            "preflight_estimate_usd": round(preflight, 6),
            "budget_usd": budget_usd,
            "ledger_total_usd": round(ledger_total, 6),
            "remaining_usd": round(budget_usd - ledger_total, 6),
            "ledger_path": str(ledger.path),
            # LLM tokens across modes, including $0 ChatGPT-subscription calls.
            "llm_tokens": llm_tokens,
        },
        "index": {"db_path": str(ecfg.db_path), "fingerprint": index_fingerprint(cfg)},
        "config": _config_summary(cfg, mode_list),
    }
    runs_dir = eval_dir() / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    report_path = runs_dir / f"{run_id}.json"
    report["report_path"] = str(report_path)
    report_path.write_text(json.dumps(report, indent=2))
    return report
