"""Integration checks on the REAL artifacts (read-only): metrics.json, the README results block,
the figures it links and the numpy Recommender.

Tests that need generated artifacts skip when they are missing (fresh clone, CI without data), so
the suite stays green before `scripts/train_all.py` + `scripts/evaluate.py` have been run.
Nothing here writes to the repo.
"""
import importlib
import json
import math
import re
from datetime import datetime
from pathlib import Path

import pytest

from cinematch.readme_fill import render_results_markdown, split_readme

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
METRICS = ROOT / "reports" / "metrics.json"
MODELS = ROOT / "models"
PROCESSED = ROOT / "data" / "processed"
FOUR_DECIMALS = re.compile(r"\d\.\d{4}")
FIGURE_REF = re.compile(r"reports/figures/[\w\-]+\.png")
SCREENSHOT = "reports/figures/app_screenshot.png"   # captured separately from the pipeline
STALE = "reports/metrics.json is older than the current pipeline: rerun scripts/train_all.py + scripts/evaluate.py"

TOP_KEYS = {"generated_at", "dataset", "split", "fast", "train_seconds", "data", "rating_models",
            "clustering", "anomaly", "bandit", "dqn"}
DATA_KEYS = {"n_users", "n_items", "n_cold_items", "n_flagged_users", "n_train", "n_val", "n_test", "n_cold",
             "n_eval_users"}
RANK_KEYS = ("precision_at_10", "recall_at_10", "ndcg_at_10")
DETECTOR_KEYS = {"epsilon", "val_f1", "test_precision", "test_recall", "test_f1", "per_bot_type_recall"}
DQN_KEYS = {"dqn", "greedy_cf", "random", "diversity_heuristic", "n_train_users", "n_eval_users", "episodes",
            "session_len"}


@pytest.fixture(scope="module")
def metrics() -> dict:
    if not METRICS.exists():
        pytest.skip("reports/metrics.json not generated yet (run scripts/evaluate.py)")
    return json.loads(METRICS.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def readme_parts() -> tuple[str, str, str]:
    return split_readme(README.read_text(encoding="utf-8"))


def _numbers(obj, path="metrics"):
    """Yield (path, value) for every number nested in a JSON object."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _numbers(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _numbers(v, f"{path}[{i}]")
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        yield path, obj


def _is_prob(x) -> bool:
    return x is not None and 0.0 <= float(x) <= 1.0


# --------------------------------------------------------------------------- metrics.json
def test_metrics_top_level_and_numbers_are_finite(metrics):
    assert TOP_KEYS <= set(metrics), TOP_KEYS - set(metrics)
    datetime.fromisoformat(metrics["generated_at"])
    assert metrics["dataset"] == "MovieLens-1M"
    bad = [(p, v) for p, v in _numbers(metrics) if not math.isfinite(v)]
    assert not bad, bad


def test_metrics_data_and_rating_models(metrics):
    d = metrics["data"]
    assert d is not None and DATA_KEYS <= set(d)
    assert d["n_users"] > 0 and d["n_train"] > d["n_test"] > 0 and d["n_eval_users"] > 0
    rm = metrics["rating_models"]
    for key in ("popularity", "bias_baseline", "cf_mf", "two_tower"):
        assert rm.get(key), f"rating_models.{key} missing"
        assert all(_is_prob(rm[key][k]) for k in RANK_KEYS), key
    assert rm["popularity"]["rmse"] is None  # popularity predicts no rating, never a fake number
    for key in ("bias_baseline", "cf_mf", "two_tower"):
        assert 0.5 < rm[key]["rmse"] < 2.0, key
    cold = rm["two_tower"].get("cold")
    assert cold and all(_is_prob(cold[k]) for k in RANK_KEYS) and cold["n_eval_users"] > 0


def test_metrics_ranking_rows_and_new_users_when_present(metrics):
    """The validation-chosen ranking rows and the new-user simulation (absent in pre-item-kNN runs)."""
    rm = metrics["rating_models"]
    if rm.get("cf_mf_pop"):
        assert all(_is_prob(rm["cf_mf_pop"][k]) for k in RANK_KEYS) and rm["cf_mf_pop"]["beta"] >= 0
    if rm.get("item_knn"):
        assert all(_is_prob(rm["item_knn"][k]) for k in RANK_KEYS)
        assert rm["item_knn"]["k"] > 0 and rm["item_knn"]["gamma"] >= 0
    nu = rm.get("new_users")
    if nu:
        assert nu["n_users"] > 0 and nu["n_seed_ratings"] > 0
        assert set(nu["strategies"]) == {"popularity", "mf_foldin", "mf_foldin_pop", "item_knn"}
        assert all(_is_prob(m[k]) for m in nu["strategies"].values() for k in RANK_KEYS)


def test_metrics_clustering_anomaly_bandit(metrics):
    c = metrics["clustering"]
    assert c and c["k"] == len(c["names"]) == len(c["sizes"]) and -1.0 <= c["silhouette"] <= 1.0
    a = metrics["anomaly"]
    assert a and DETECTOR_KEYS <= set(a)
    assert all(_is_prob(a[k]) for k in ("val_f1", "test_precision", "test_recall", "test_f1"))
    # Deployed model = 5 behaviour features; the ablation adds the item-deviation feature.
    assert "ablation_plus_item_dev" in a and "ablation_5_features" not in a, STALE
    assert DETECTOR_KEYS <= set(a["ablation_plus_item_dev"])
    b = metrics["bandit"]
    assert b and set(b["policies"]) == {"random", "epsilon_greedy", "linucb"}
    for p in b["policies"].values():
        assert _is_prob(p["mean_reward"]) and p["cum_regret"] >= 0 and p["cum_regret_std"] >= 0
    assert all(p["mean_reward"] <= b["oracle_mean_reward"] + 1e-9 for p in b["policies"].values())
    # Hyperparameters are tuned on disjoint users and the tuning grid is reported.
    params = b.get("params") or {}
    assert {"alpha", "epsilon", "ridge", "tuned_on_users"} <= set(params), STALE
    assert params["tuned_on_users"] > 0
    tuning = b.get("tuning") or {}
    assert tuning.get("alpha") and tuning.get("epsilon"), STALE
    assert _is_prob(b.get("frac_users_no_signal"))


def test_metrics_dqn_is_null_or_complete(metrics):
    d = metrics["dqn"]
    if d is None:
        return
    assert DQN_KEYS <= set(d), DQN_KEYS - set(d)
    assert d["n_train_users"] > 0 and d["n_eval_users"] > 0 and d["session_len"] == 5


# --------------------------------------------------------------------------- README
def test_readme_block_matches_metrics(metrics, readme_parts):
    """The block is exactly what the current renderer makes of metrics.json (no hand edits, no stale render)."""
    _, block, _ = readme_parts
    expected = render_results_markdown(metrics).strip()
    assert block.strip() == expected, "README results block is stale: rerun scripts/evaluate.py"


def test_readme_has_no_hand_typed_metrics(readme_parts):
    head, _, tail = readme_parts
    assert not FOUR_DECIMALS.findall(head + tail), FOUR_DECIMALS.findall(head + tail)


def test_readme_figures_exist(metrics):
    """Every figure the README links (inside and outside the markers) exists on disk."""
    refs = sorted(set(FIGURE_REF.findall(README.read_text(encoding="utf-8"))) - {SCREENSHOT})
    assert refs, "README links no figures"
    missing = [r for r in refs if not (ROOT / r).exists()]
    assert not missing, missing


def test_readme_screenshot_present():
    assert SCREENSHOT in README.read_text(encoding="utf-8")
    if not (ROOT / SCREENSHOT).exists():
        pytest.skip(f"{SCREENSHOT} not captured yet")


def test_concepts_table_names_exist(readme_parts):
    """Function/class names in the repository-structure table really exist in the named modules."""
    _, _, tail = readme_parts
    section = tail.split("## Repository structure")[1].split("\n## ")[0]
    rows = [r for r in section.splitlines() if "src/cinematch/" in r]
    assert len(rows) >= 6
    for row in rows:
        mod_name = re.search(r"src/cinematch/(\w+)\.py", row).group(1)
        mod = importlib.import_module(f"cinematch.{mod_name}")
        for name in re.findall(r"`([A-Za-z_][\w.]*)`", row.split(".py`", 1)[1]):
            obj = mod
            for part in name.split("."):
                assert hasattr(obj, part), f"cinematch.{mod_name} has no {name}"
                obj = getattr(obj, part)


# --------------------------------------------------------------------------- serving on real models
def test_recommender_serves_real_models():
    try:
        from cinematch.recommend import ModelsMissingError, Recommender
    except ImportError:
        pytest.skip("cinematch.recommend not available")
    if not (MODELS / "item_knn.npz").exists():
        pytest.skip("models predate item-kNN (run scripts/train_all.py)")
    try:
        rec = Recommender(models_dir=MODELS, processed_dir=PROCESSED)
    except ModelsMissingError:
        pytest.skip("real models not trained yet (run scripts/train_all.py)")
    popular = rec.popular_for_rating(10)
    rated = {int(m): 5.0 for m in popular["movie_id"].head(5)}
    for strategy in ("knn", "cf", "content"):
        out = rec.recommend_from_ratings(rated, strategy=strategy, n=10)
        assert len(out) == 10, strategy
        assert not set(out["movie_id"].astype(int)) & set(rated), strategy  # never re-recommend rated movies
        assert out["score"].notna().all(), strategy
    for cid in range(rec.k):
        assert len(rec.recommend_for_cluster(cid, n=10)) == 10
    assert len(rec.recommend_strategy(0, "cf", n=10)) == 10
