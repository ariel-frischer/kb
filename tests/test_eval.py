"""Tests for kb.eval — BEIR loading, retrieval metrics, spend cap, index isolation."""

import gzip
import json
import math
import urllib.error
import zipfile
from unittest.mock import MagicMock, patch

import pytest

import kb.eval as kb_eval
from kb.config import Config
from kb.eval import (
    EvalBudgetError,
    SpendLedger,
    collapse_to_docs,
    ensure_dataset,
    eval_config,
    extract_beir_zip,
    load_beir,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    run_eval,
    safe_doc_id,
)

CORPUS = [
    {
        "_id": "d1",
        "title": "Zebrafish fin regeneration",
        "text": "Zebrafish regenerate amputated fins through blastema formation "
        "and osteoblast dedifferentiation.",
    },
    {
        "_id": "d2",
        "title": "Axolotl limb regrowth",
        "text": "Axolotl salamanders regrow entire limbs; zebrafish fins show a "
        "related regeneration program.",
    },
    {
        "_id": "doc/3.v2",
        "title": "Graphene conductivity",
        "text": "Graphene sheets exhibit extremely high electron mobility and "
        "thermal conductivity at room temperature.",
    },
    {
        "_id": "d4",
        "title": "Coral bleaching",
        "text": "Ocean warming expels symbiotic algae from coral tissue, causing "
        "widespread coral bleaching events.",
    },
    {
        "_id": "d5",
        "title": "Tardigrade survival",
        "text": "Tardigrades survive desiccation and vacuum exposure using "
        "protective disordered proteins.",
    },
]
QUERIES = [
    {"_id": "q1", "text": "zebrafish fin regeneration"},
    {"_id": "q2", "text": "graphene electron mobility"},
    {"_id": "q3", "text": "coral bleaching ocean warming"},
    {"_id": "q4", "text": "tardigrade desiccation"},
    {"_id": "q5", "text": "axolotl limbs"},
    {"_id": "q6", "text": "query with only non-relevant judgments"},
]
QRELS = (
    "query-id\tcorpus-id\tscore\n"
    "q1\td1\t2\n"
    "q1\td2\t1\n"
    "q2\tdoc/3.v2\t1\n"
    "q3\td4\t1\n"
    "q4\td5\t1\n"
    "q5\td2\t1\n"
    "q6\td1\t0\n"
)


def _jsonl(rows: list[dict]) -> str:
    return "".join(json.dumps(r) + "\n" for r in rows)


def _write_fixture_zip(path, name="scifact"):
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(f"{name}/corpus.jsonl", _jsonl(CORPUS))
        zf.writestr(f"{name}/queries.jsonl", _jsonl(QUERIES))
        zf.writestr(f"{name}/qrels/test.tsv", QRELS)
        zf.writestr(f"{name}/qrels/train.tsv", "query-id\tcorpus-id\tscore\n")
    return path


@pytest.fixture
def eval_home(tmp_path, monkeypatch):
    """Redirect GLOBAL_DATA_DIR/eval to a temp dir; keep chunking offline."""
    home = tmp_path / "eval"
    monkeypatch.setattr(kb_eval, "EVAL_DIR", home)
    # chonkie's markdown recipe is fetched from the Hugging Face Hub; use regex chunking.
    monkeypatch.setattr("kb.chunk.CHONKIE_AVAILABLE", False)
    return home


@pytest.fixture
def installed_dataset(eval_home, tmp_path):
    """Fixture BEIR dataset extracted where ensure_dataset expects it (no network)."""
    zip_path = _write_fixture_zip(tmp_path / "fixture.zip")
    extract_beir_zip(zip_path, kb_eval.dataset_dir("scifact") / "raw")
    return "scifact"


@pytest.fixture
def user_cfg(tmp_path):
    """User config with a priced API embedding model (mocked) and no LLM steps."""
    cfg = Config(embed_dims=4, hyde_enabled=False, query_expand=False)
    cfg.scope = "project"
    cfg.config_dir = tmp_path / "project"
    cfg.config_path = tmp_path / "project" / ".kb.toml"
    cfg.db_path = tmp_path / "user-data" / "kb.db"
    return cfg


def _embedding_client(dims=4):
    client = MagicMock()

    def create(model, input, dimensions):
        resp = MagicMock()
        resp.data = [MagicMock(embedding=[0.1] * dims) for _ in input]
        return resp

    client.embeddings.create.side_effect = create
    return client


def _fake_search(cost_usd):
    """search_core stand-in: returns the query's first qrel doc, charging cost_usd."""
    targets = {q["text"]: None for q in QUERIES}
    for line in QRELS.splitlines()[1:]:
        qid, doc_id, _ = line.split("\t")
        text = next(q["text"] for q in QUERIES if q["_id"] == qid)
        targets[text] = targets[text] or doc_id

    def search(query, cfg, top_k=5, threshold=None):
        doc_id = targets.get(query) or "d1"
        return {
            "results": [
                {
                    "doc_path": f"{safe_doc_id(doc_id)}.md",
                    "heading": None,
                    "text": "x",
                }
            ],
            "cost": {"estimated_total_usd": cost_usd},
        }

    return search


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


class TestMetrics:
    def test_ndcg_uses_graded_relevance(self):
        rels = {"d1": 2, "d2": 1}
        swapped = ndcg_at_k(["d2", "d1"], rels)
        expected_dcg = 1 / math.log2(2) + 2 / math.log2(3)
        ideal_dcg = 2 / math.log2(2) + 1 / math.log2(3)
        assert swapped == pytest.approx(expected_dcg / ideal_dcg)
        assert ndcg_at_k(["d1", "d2"], rels) == pytest.approx(1.0)
        assert swapped < 1.0

    def test_ndcg_ignores_hits_below_cutoff(self):
        ranked = [f"x{i}" for i in range(10)] + ["d1"]
        assert ndcg_at_k(ranked, {"d1": 1}) == 0.0

    def test_recall_mrr_precision(self):
        rels = {"a": 1, "b": 1, "c": 1, "d": 1}
        ranked = ["x", "y", "a", "b"]
        assert recall_at_k(ranked, rels) == 0.5
        assert mrr_at_k(ranked, rels) == pytest.approx(1 / 3)
        assert precision_at_k(ranked, rels) == pytest.approx(0.2)

    def test_mrr_zero_without_relevant_hit(self):
        assert mrr_at_k(["x", "y"], {"a": 1}) == 0.0

    def test_collapse_keeps_first_rank_per_doc(self):
        paths = ["b.md", "a.md", "b.md", None, "c.md", "a.md"]
        assert collapse_to_docs(paths, {}) == ["b", "a", "c"]

    def test_collapse_maps_safe_ids_excludes_query_and_truncates(self):
        weird = "doc/3.v2"
        id_map = {safe_doc_id(weird): weird}
        paths = [f"{safe_doc_id(weird)}.md", "q1.md", "d1.md", "d2.md"]
        assert collapse_to_docs(paths, id_map, exclude="q1", k=2) == [weird, "d1"]


class TestSafeDocId:
    def test_plain_ids_unchanged(self):
        assert safe_doc_id("31715818") == "31715818"

    def test_unsafe_ids_do_not_collide(self):
        assert safe_doc_id("a/b") != safe_doc_id("a_b")
        assert "/" not in safe_doc_id("a/b")


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


class TestBeirLoader:
    def test_zip_layout_parsed(self, tmp_path):
        raw = tmp_path / "raw"
        extract_beir_zip(_write_fixture_zip(tmp_path / "ds.zip"), raw)
        data = load_beir(raw)

        assert data.corpus["d1"][0] == "Zebrafish fin regeneration"
        assert "doc/3.v2" in data.corpus
        assert data.qrels["q1"] == {"d1": 2, "d2": 1}
        # q6 has only zero-score judgments: not an evaluable query
        assert "q6" not in data.qrels
        assert set(data.queries) == {"q1", "q2", "q3", "q4", "q5"}

    def test_zip_missing_member_rejected(self, tmp_path):
        bad = tmp_path / "bad.zip"
        with zipfile.ZipFile(bad, "w") as zf:
            zf.writestr("x/corpus.jsonl", "")
        with pytest.raises(kb_eval.EvalError, match="queries.jsonl"):
            extract_beir_zip(bad, tmp_path / "raw")

    def test_ensure_dataset_downloads_ukp_zip(self, eval_home, tmp_path, monkeypatch):
        fixture = _write_fixture_zip(tmp_path / "fixture.zip")
        urls = []

        def fake_download(url, dest, timeout=120):
            urls.append(url)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(fixture.read_bytes())

        monkeypatch.setattr(kb_eval, "_download", fake_download)
        raw = ensure_dataset("scifact")
        assert urls == [kb_eval.UKP_URL.format(name="scifact")]
        assert len(load_beir(raw).queries) == 5
        # Cached: no second download
        ensure_dataset("scifact")
        assert len(urls) == 1

    def test_ensure_dataset_falls_back_to_hf_mirror(self, eval_home, monkeypatch):
        payloads = {
            "corpus.jsonl.gz": gzip.compress(_jsonl(CORPUS).encode()),
            "queries.jsonl.gz": gzip.compress(_jsonl(QUERIES).encode()),
            "test.tsv": QRELS.encode(),
        }
        urls = []

        def fake_download(url, dest, timeout=120):
            urls.append(url)
            if "ukp" in url:
                raise urllib.error.URLError("down")
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(payloads[url.rsplit("/", 1)[1]])

        monkeypatch.setattr(kb_eval, "_download", fake_download)
        data = load_beir(ensure_dataset("scifact"))
        assert data.qrels["q2"] == {"doc/3.v2": 1}
        rev, qrels_rev = kb_eval.DATASETS["scifact"]
        assert (
            f"https://huggingface.co/datasets/BeIR/scifact/resolve/{rev}/corpus.jsonl.gz"
            in urls
        )
        assert (
            f"https://huggingface.co/datasets/BeIR/scifact-qrels/resolve/{qrels_rev}/test.tsv"
            in urls
        )


# ---------------------------------------------------------------------------
# Isolation + end-to-end FTS
# ---------------------------------------------------------------------------


class TestIsolation:
    def test_eval_db_path_is_isolated(self, eval_home, user_cfg, tmp_path):
        ecfg = eval_config(user_cfg, "scifact", tmp_path / "corpus")
        assert ecfg.db_path != user_cfg.db_path
        assert ecfg.db_path.is_relative_to(eval_home / "indexes")
        assert ecfg.config_path is None
        # User config object untouched
        assert user_cfg.db_path == tmp_path / "user-data" / "kb.db"

    def test_index_fingerprint_changes_with_embedding_model(self, user_cfg):
        other = Config(**{**user_cfg.__dict__, "embed_model": "text-embedding-3-large"})
        assert kb_eval.index_fingerprint(user_cfg) != kb_eval.index_fingerprint(other)

    def test_fts_run_scores_real_index_without_touching_user_db(
        self, installed_dataset, eval_home, user_cfg
    ):
        with patch("kb.ingest.OpenAI", return_value=_embedding_client()):
            report = run_eval(installed_dataset, user_cfg, modes="fts", budget=1.0)

        assert not user_cfg.db_path.exists()
        assert report["index"]["db_path"].startswith(str(eval_home))
        fts = report["metrics"]["fts"]
        assert fts["queries"] == 5
        assert fts["ndcg@10"] > 0.8
        assert fts["usd"] == 0.0
        assert report["spend"]["index_usd"] > 0
        assert (eval_home / "runs" / f"{report['run_id']}.json").is_file()

    def test_rerun_reuses_index_and_spends_nothing_on_embeddings(
        self, installed_dataset, eval_home, user_cfg
    ):
        client = _embedding_client()
        with patch("kb.ingest.OpenAI", return_value=client):
            run_eval(installed_dataset, user_cfg, modes="fts", budget=1.0)
            calls = client.embeddings.create.call_count
            report = run_eval(installed_dataset, user_cfg, modes="fts", budget=1.0)
        assert client.embeddings.create.call_count == calls
        assert report["spend"]["run_usd"] == 0.0


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------


class TestBudget:
    def test_preflight_refuses_before_any_api_call(
        self, installed_dataset, eval_home, user_cfg
    ):
        client = _embedding_client()
        with patch("kb.ingest.OpenAI", return_value=client):
            with pytest.raises(EvalBudgetError, match="exceeds the remaining"):
                run_eval(installed_dataset, user_cfg, modes="fts", budget=1e-9)
        client.embeddings.create.assert_not_called()
        assert not (eval_home / "indexes").exists()
        assert SpendLedger(kb_eval.ledger_path()).total() == 0.0

    def test_unpriced_api_model_refused_before_download(
        self, eval_home, user_cfg, monkeypatch
    ):
        user_cfg.embed_model = "nomic-embed-text"

        def no_download(*args, **kwargs):
            raise AssertionError("must refuse before downloading")

        monkeypatch.setattr(kb_eval, "_download", no_download)
        with pytest.raises(EvalBudgetError, match="nomic-embed-text"):
            run_eval("scifact", user_cfg, modes="fts")

    def test_unpriced_rerank_model_refused(self, eval_home, user_cfg):
        user_cfg.embed_method = "local"
        user_cfg.chat_model = "some-unpriced-llm"
        with pytest.raises(EvalBudgetError, match="chat_model=some-unpriced-llm"):
            run_eval("scifact", user_cfg, modes="rerank")

    def test_local_pipeline_needs_no_budget(self, user_cfg):
        user_cfg.embed_method = "local"
        user_cfg.rerank_method = "cross-encoder"
        usage = kb_eval.api_usage(user_cfg, ["fts", "hybrid", "rerank"])
        assert not usage.any
        assert (
            kb_eval.estimate_query_cost_usd("q", user_cfg, usage, ["hybrid", "rerank"])
            == 0.0
        )

    def test_mid_run_stop_reports_partial_and_stays_under_budget(
        self, installed_dataset, eval_home, user_cfg, monkeypatch
    ):
        # Per-query estimates undershoot actual spend: the runtime guard must
        # still stop before the cumulative ledger crosses the budget.
        monkeypatch.setattr(kb_eval, "estimate_query_cost_usd", lambda *a: 0.01)
        monkeypatch.setattr(kb_eval, "search_core", _fake_search(0.3))
        with patch("kb.ingest.OpenAI", return_value=_embedding_client()):
            report = run_eval(installed_dataset, user_cfg, modes="hybrid", budget=1.0)

        assert report["budget_exhausted"] is True
        assert report["metrics"]["hybrid"]["queries"] == 3
        assert report["metrics"]["hybrid"]["ndcg@10"] > 0
        ledger_total = SpendLedger(kb_eval.ledger_path()).total()
        assert ledger_total == pytest.approx(0.9 + report["spend"]["index_usd"])
        assert ledger_total <= 1.0

    def test_ledger_accumulates_across_runs(
        self, installed_dataset, eval_home, user_cfg, monkeypatch
    ):
        monkeypatch.setattr(kb_eval, "search_core", _fake_search(0.25))
        with patch("kb.ingest.OpenAI", return_value=_embedding_client()):
            first = run_eval(
                installed_dataset, user_cfg, modes="hybrid", limit=2, budget=10.0
            )
            second = run_eval(
                installed_dataset, user_cfg, modes="hybrid", limit=2, budget=10.0
            )
            assert second["spend"]["ledger_total_usd"] == pytest.approx(
                first["spend"]["ledger_total_usd"] + 0.5
            )
            # Prior spend (> $1) leaves nothing for a $1 budget: refused up front.
            with pytest.raises(EvalBudgetError):
                run_eval(
                    installed_dataset, user_cfg, modes="hybrid", limit=1, budget=1.0
                )

        on_disk = json.loads(kb_eval.ledger_path().read_text())
        assert on_disk["total_usd"] == pytest.approx(
            second["spend"]["ledger_total_usd"]
        )
        assert {first["run_id"], second["run_id"]} <= set(on_disk["runs"])

    def test_bad_mode_rejected(self, user_cfg):
        with pytest.raises(kb_eval.EvalError, match="bogus"):
            run_eval("scifact", user_cfg, modes="fts,bogus")
