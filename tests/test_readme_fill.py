"""Tests for metrics assembly, README rendering and the number-provenance rule."""
import json
import re
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from cinematch.readme_fill import (  # noqa: E402
    MARKER_END,
    MARKER_START,
    assemble_metrics,
    fmt,
    render_results_markdown,
    split_readme,
    update_readme,
)

FOUR_DECIMALS = re.compile(r"\d\.\d{4}")


def _cfg(fast=False):
    return types.SimpleNamespace(seed=42, fast=fast, cold_item_frac=0.06)


def _rank(rmse=None):
    return {"rmse": rmse, "precision_at_10": 0.1234, "recall_at_10": 0.05, "ndcg_at_10": 0.15}


def _full_outputs():
    """Fake eval_step outputs for every step (shapes of section 6.3)."""
    return {
        "data": {"n_users": 6040, "n_items": 3883, "n_cold_items": 120, "n_flagged_users": 30,
                 "n_train": 790000, "n_val": 98000, "n_test": 98000, "n_cold": 14000, "n_eval_users": 2000},
        "cf": {"popularity": _rank(None), "bias_baseline": _rank(0.91),
               "cf_mf": {**_rank(0.87), "best_epoch": 7, "dim": 32}},
        "content": {"two_tower": {**_rank(0.95), "best_epoch": 4,
                                  "cold": {**_rank(1.01), "n_eval_users": 800}}},
        "clustering": {"k": 3, "silhouette": 0.0712, "names": ["Horror & Thriller fans", "Drama & War fans", "Mainstream (Comedy)"],
                       "sizes": [1500, 2000, 2510]},
        "anomaly": {"epsilon": -12.5, "epsilon_capped": False, "val_f1": 0.9, "test_precision": 0.95,
                    "test_recall": 0.88, "test_f1": 0.913, "n_bots_test": 150, "n_real_flagged": 5,
                    "n_real_flagged_all": 30, "per_bot_type_recall": {"push": 1.0, "camouflage_slow": 0.6}},
        "bandit": {"rounds": 3000, "n_seeds": 5,
                   "policies": {"random": {"mean_reward": 0.05, "cum_regret": 120.0, "cum_regret_std": 4.0},
                                "epsilon_greedy": {"mean_reward": 0.07, "cum_regret": 60.0, "cum_regret_std": 5.0},
                                "linucb": {"mean_reward": 0.08, "cum_regret": 30.0, "cum_regret_std": 3.0}},
                   "oracle_mean_reward": 0.09,
                   "arm_mean_expected_reward": {"cf": 0.08, "content": 0.05, "cluster_popular": 0.06, "trending": 0.04}},
        "dqn": None,
    }


# --------------------------------------------------------------------------- assembly
def test_assemble_merges_rating_models_and_sections():
    m = assemble_metrics(_full_outputs(), _cfg(), {"data": 12.3, "cf": 80.0, "total": 92.3}, generated_at="2026-01-01T00:00:00+00:00")
    assert set(m["rating_models"]) == {"popularity", "bias_baseline", "cf_mf", "two_tower"}
    assert m["clustering"]["k"] == 3 and m["bandit"]["rounds"] == 3000
    assert m["dqn"] is None
    assert m["dataset"] == "MovieLens-1M"
    assert m["split"] == "per-user time-aware 80/10/10, 6% cold items, seed 42"
    assert m["fast"] is False
    assert m["train_seconds"] == {"anomaly": None, "data": 12.3, "cf": 80.0, "content": None,
                                  "clustering": None, "bandit": None, "dqn": None}
    json.dumps(m)  # serialisable


def test_assemble_skipped_steps_are_null():
    m = assemble_metrics({"data": {"n_users": 10}, "cf": None}, _cfg(fast=True), None)
    assert m["rating_models"] == {}
    for key in ("clustering", "anomaly", "bandit", "dqn"):
        assert m[key] is None
    assert m["data"] == {"n_users": 10}
    assert m["fast"] is True
    assert all(v is None for v in m["train_seconds"].values())


def test_assemble_cleans_numpy_rounds_and_drops_nan():
    out = {"clustering": {"k": np.int64(4), "silhouette": np.float32(0.123456789),
                          "sizes": np.array([1, 2, 3]), "flag": np.bool_(True), "bad": float("nan")}}
    m = assemble_metrics(out, _cfg(), {})
    c = m["clustering"]
    assert c["k"] == 4 and type(c["k"]) is int
    assert c["silhouette"] == 0.1235
    assert c["sizes"] == [1, 2, 3]
    assert c["flag"] is True
    assert c["bad"] is None
    json.dumps(m)


def test_assemble_cleans_pandas_missing_and_rejects_unknown_types():
    pd = pytest.importorskip("pandas")
    m = assemble_metrics({"data": {"a": pd.NA, "b": pd.NaT, "p": Path("x") / "y"}}, _cfg(), {})
    assert m["data"] == {"a": None, "b": None, "p": "x/y"}
    with pytest.raises(TypeError, match="object"):
        assemble_metrics({"data": {"bad": object()}}, _cfg(), {})


# --------------------------------------------------------------------------- rendering
def test_render_full_metrics_has_4_decimals_and_sections():
    m = assemble_metrics(_full_outputs(), _cfg(), {"cf": 80.0}, generated_at="2026-01-01T00:00:00+00:00")
    text = render_results_markdown(m)
    assert "| Matrix factorization (CF) | 0.8700 | 0.1234 | 0.0500 | 0.1500 |" in text
    assert "| Popularity | n/a | 0.1234" in text
    assert "cold-start" in text and "1.0100" in text
    assert "Horror & Thriller fans (1,500)" in text
    assert "| LinUCB | 0.0800 | 30.0000 ± 3.0000 |" in text
    assert "DQN stretch goal:** not shipped" in text
    assert "2026-01-01T00:00:00+00:00" in text
    assert "reports/figures/cf_training.png" in text
    assert "FAST MODE" not in text


def test_render_nulls_as_na_and_not_shipped():
    m = assemble_metrics({}, _cfg(), None, generated_at="x")
    text = render_results_markdown(m)
    assert "n/a" in text and "not shipped" in text
    assert not FOUR_DECIMALS.search(text)  # no numbers invented for missing models
    assert "reports/figures" not in text   # no links to figures that were never drawn


def test_render_fast_banner():
    text = render_results_markdown(assemble_metrics({}, _cfg(fast=True), None))
    assert "FAST MODE" in text


def test_fmt():
    assert fmt(None) == "n/a"
    assert fmt(0.5) == "0.5000"
    assert fmt(np.float32(0.12345678)) == "0.1235"


# --------------------------------------------------------------------------- README update
def _readme(tmp_path, body="old block"):
    p = tmp_path / "README.md"
    p.write_text(f"# Title\n\nMentions `{MARKER_START}` inline.\n\n{MARKER_START}\n{body}\n{MARKER_END}\n\n## After\ntext\n",
                 encoding="utf-8")
    return p


def test_update_readme_replaces_block_and_is_idempotent(tmp_path):
    p = _readme(tmp_path)
    update_readme(p, "NEW 1.2345")
    once = p.read_text(encoding="utf-8")
    update_readme(p, "NEW 1.2345")
    assert p.read_text(encoding="utf-8") == once
    head, block, tail = split_readme(once)
    assert block.strip() == "NEW 1.2345"
    assert "old block" not in once
    assert head.startswith("# Title") and "inline" in head
    assert tail.startswith(MARKER_END) and "## After\ntext" in tail


def test_update_readme_missing_markers_raises(tmp_path):
    p = tmp_path / "README.md"
    p.write_text(f"# no markers here, only inline `{MARKER_START}`\n", encoding="utf-8")
    with pytest.raises(ValueError):
        update_readme(p, "x")
    p.write_text(f"{MARKER_END}\nbefore start\n{MARKER_START}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        update_readme(p, "x")


def test_repo_readme_has_no_hand_typed_metrics():
    """Number-provenance rule: 4-decimal numbers may only appear inside the markers."""
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    head, _, tail = split_readme(text)
    outside = head + tail
    assert not FOUR_DECIMALS.findall(outside), FOUR_DECIMALS.findall(outside)


# --------------------------------------------------------------------------- step runner
def test_step_runner_skips_missing_and_records_outputs(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    import train_all  # a broken import must fail, not skip
    monkeypatch.setattr(train_all, "set_seed", lambda seed: None)  # avoid loading torch

    fake = types.ModuleType("cinematch.fake_step")
    fake.eval_step = lambda cfg, ctx: {"k": 1}
    no_fn = types.ModuleType("cinematch.fake_nofn")
    monkeypatch.setitem(sys.modules, "cinematch.fake_step", fake)
    monkeypatch.setitem(sys.modules, "cinematch.fake_nofn", no_fn)

    ctx = train_all.run(["fake_step", "fake_nofn", "does_not_exist_xyz"], "eval_step", _cfg())
    assert ctx["outputs"] == {"fake_step": {"k": 1}}
    assert set(ctx["timings"]) == {"fake_step"}
    m = assemble_metrics({"clustering": ctx["outputs"]["fake_step"]}, _cfg(), ctx["timings"])
    assert m["clustering"] == {"k": 1}


# --------------------------------------------------------------------------- scripts end-to-end
def _patch_scripts(monkeypatch, tmp_path, order):
    """Import train_all/evaluate with every repo path redirected into tmp_path."""
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    monkeypatch.delenv("CINEMATCH_FAST", raising=False)
    import evaluate
    import train_all

    fake = types.ModuleType("cinematch.fake_e2e")
    fake.train_step = lambda cfg, ctx: None
    fake.eval_step = lambda cfg, ctx: {"n_users": 10}
    monkeypatch.setitem(sys.modules, "cinematch.fake_e2e", fake)
    monkeypatch.setattr(train_all, "set_seed", lambda seed: None)  # avoid loading torch
    monkeypatch.setattr(train_all, "ensure_dirs", lambda: None)
    monkeypatch.setattr(train_all, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(train_all, "TRAIN_ORDER", order)
    monkeypatch.setattr(evaluate, "ensure_dirs", lambda: None)
    monkeypatch.setattr(evaluate, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(evaluate, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(evaluate, "README_PATH", _readme(tmp_path))
    monkeypatch.setattr(evaluate, "EVAL_ORDER", order)
    return train_all, evaluate


def test_train_all_writes_timings(monkeypatch, tmp_path):
    train_all, _ = _patch_scripts(monkeypatch, tmp_path, ["fake_e2e", "missing_step_xyz"])
    train_all.main(["--fast"])
    timings = json.loads((tmp_path / "train_timings.json").read_text(encoding="utf-8"))
    assert set(timings) == {"fake_e2e", "total", "fast"}
    assert timings["fast"] is True


def test_evaluate_writes_null_metrics_and_readme(monkeypatch, tmp_path):
    _, evaluate = _patch_scripts(monkeypatch, tmp_path, ["missing_step_xyz"])
    evaluate.main(["--fast"])
    m = json.loads((tmp_path / "reports" / "metrics.json").read_text(encoding="utf-8"))
    assert set(m) == {"generated_at", "dataset", "split", "fast", "train_seconds", "data",
                      "rating_models", "clustering", "anomaly", "bandit", "dqn"}
    assert m["fast"] is True and m["rating_models"] == {}
    assert all(m[k] is None for k in ("data", "clustering", "anomaly", "bandit", "dqn"))
    assert all(v is None for v in m["train_seconds"].values())
    _, block, _ = split_readme(evaluate.README_PATH.read_text(encoding="utf-8"))
    assert "FAST MODE" in block and "not shipped" in block and m["generated_at"] in block


def test_evaluate_flags_fast_trained_models(monkeypatch, tmp_path):
    train_all, evaluate = _patch_scripts(monkeypatch, tmp_path, ["fake_e2e"])
    train_all.main(["--fast"])
    m = evaluate.main([])  # full evaluate on models trained with --fast
    assert m["fast"] is True
    assert m["fake_e2e"] == {"n_users": 10}
    assert "FAST MODE" in evaluate.README_PATH.read_text(encoding="utf-8")
