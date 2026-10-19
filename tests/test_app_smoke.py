"""Smoke tests for the Streamlit app (streamlit.testing.v1.AppTest).

The "models present" tests read the real repo artifacts (read-only) and skip when they are missing;
the "models missing" branch is exercised by monkeypatching the Recommender to raise.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

st = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = ROOT / "app" / "streamlit_app.py"
HAS_RECOMMEND = importlib.util.find_spec("cinematch.recommend") is not None
HAS_REAL_ARTIFACTS = (ROOT / "models" / "cf_mf.npz").exists() and (
    ROOT / "data" / "processed" / "ratings_split.parquet").exists()
HAS_BANDIT_ENV = (ROOT / "models" / "bandit_env.npz").exists()
TIMEOUT = 60

needs_recommend = pytest.mark.skipif(not HAS_RECOMMEND, reason="cinematch.recommend not implemented")
needs_models = pytest.mark.skipif(not (HAS_RECOMMEND and HAS_REAL_ARTIFACTS),
                                  reason="real models/processed data missing (run scripts/train_all.py)")


@pytest.fixture(autouse=True)
def _fresh_caches():
    """st.cache_* is process-global: clear it so tests cannot see each other's loaded models."""
    st.cache_resource.clear()
    st.cache_data.clear()
    yield
    st.cache_resource.clear()
    st.cache_data.clear()


def _run() -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=TIMEOUT)
    at.run(timeout=TIMEOUT)
    return at


def _button(at: AppTest, label: str):
    matches = [b for b in at.button if b.label == label]
    assert matches, f"button {label!r} not found"
    return matches[0]


@needs_models
def test_app_renders_all_tabs():
    at = _run()
    assert not at.exception
    assert [t.label for t in at.tabs] == ["Get Recommendations", "Explore Taste Profiles", "Bandit Lab",
                                          "How It Works"]
    assert not any("Models not found" in w.value for w in at.warning)
    assert len(_rating_sliders(at)) == 30       # rating grid


def _rating_sliders(at: AppTest) -> list:
    return [s for s in at.select_slider if str(s.key).startswith("rate_")]


@needs_models
@pytest.mark.parametrize("strategy", ["Collaborative filtering (MF)", "Content (two-tower)", "Hybrid"])
def test_rate_movies_and_recommend(strategy):
    at = _run()
    for slider, value in zip(_rating_sliders(at)[:4], ["5", "4", "2", "5"]):
        slider.set_value(value)
    at.radio(key="strategy").set_value(strategy)
    _button(at, "Recommend").click()
    at.run(timeout=TIMEOUT)
    assert not at.exception
    assert any("You look like" in s.value for s in at.success)
    recs = at.session_state["recs"]
    assert len(recs["df"]) == 10
    assert recs["df"]["movie_id"].is_unique
    cards = " ".join(m.value for m in at.markdown)
    if strategy == "Hybrid":     # hybrid score is a z-score: never presented as a rating
        assert "relative score" in cards and "predicted rating" not in cards
    else:
        assert "predicted rating" in cards and "relative score" not in cards


@needs_models
def test_pick_taste_profile():
    at = _run()
    at.radio(key="rec_mode").set_value("Pick a taste profile").run(timeout=TIMEOUT)
    assert not at.exception
    assert at.selectbox(key="profile_pick") is not None


@needs_models
def test_too_few_ratings_warns():
    at = _run()
    _rating_sliders(at)[0].set_value("4")
    _button(at, "Recommend").click()
    at.run(timeout=TIMEOUT)
    assert not at.exception
    assert any("at least 3" in w.value for w in at.warning)


@needs_models
@pytest.mark.skipif(not HAS_BANDIT_ENV, reason="models/bandit_env.npz missing")
def test_bandit_lab_simulation():
    """Real env: the sliders default to bandit_results.json["params"] (config if absent) and Run works."""
    import json

    from cinematch.config import get_config

    cfg = get_config()
    results_path = ROOT / "models" / "bandit_results.json"
    params = (json.loads(results_path.read_text(encoding="utf-8")).get("params") or {}
              if results_path.exists() else {})
    at = _run()
    assert at.slider(key="bandit_epsilon").value == pytest.approx(params.get("epsilon", cfg.bandit_epsilon))
    assert at.select_slider(key="bandit_alpha").value == pytest.approx(params.get("alpha", cfg.bandit_alpha))
    assert any(c.value.startswith("Defaults ") for c in at.caption)
    at.slider(key="bandit_rounds").set_value(200)
    _button(at, "Run simulation").click()
    at.run(timeout=TIMEOUT)
    assert not at.exception
    assert set(at.session_state["bandit_results"]) == {"Random", "Epsilon-greedy", "LinUCB"}


@needs_recommend
def test_models_missing_shows_instructions(monkeypatch):
    import cinematch.recommend as recommend

    def _missing(*args, **kwargs):
        raise recommend.ModelsMissingError(["models/cf_mf.npz"])

    monkeypatch.setattr(recommend, "Recommender", _missing)
    at = _run()
    assert not at.exception
    assert any("Models not found" in w.value for w in at.warning)
    assert any("train_all.py" in c.value for c in at.code)


def _write_bandit_env(models_dir: Path) -> None:
    """A tiny synthetic simulator (30 users, arm 2 clearly best) in models_dir/bandit_env.npz."""
    import numpy as np

    from cinematch import bandit

    rng = np.random.default_rng(0)
    n_users, n_arms = 30, len(bandit.ARMS)
    probs = rng.uniform(0.0, 0.3, size=(n_users, n_arms, 10)).astype(np.float32)
    probs[:, 2] += 0.4
    contexts = np.hstack([np.ones((n_users, 1)), rng.normal(size=(n_users, 3))]).astype(np.float32)
    bandit.save_env({"user_idx": np.arange(n_users), "probs": probs, "expected": probs.mean(axis=2),
                     "contexts": contexts, "arm_names": np.array(bandit.ARMS, dtype="U20"),
                     "lists": np.zeros((n_users, n_arms, 10), dtype=np.int32)}, models_dir / "bandit_env.npz")


needs_bandit = pytest.mark.skipif(importlib.util.find_spec("cinematch.bandit") is None,
                                  reason="cinematch.bandit missing")


@needs_recommend
@needs_bandit
@pytest.mark.parametrize("params, eps, alpha, note", [
    ({"epsilon": 0.05, "alpha": 0.03, "ridge": 1.0, "tuned_on_users": 1000}, 0.05, 0.03,
     "1,000 simulated users disjoint"),
    ({"epsilon": 0.2, "alpha": 0.3, "ridge": 1.0}, 0.2, 0.3, "values used in the last training run"),
    (None, None, None, "Defaults come from the config"),           # old bandit_results.json, no "params"
])
def test_bandit_lab_defaults_from_results(tmp_path, monkeypatch, params, eps, alpha, note):
    import json

    import cinematch.config as config

    _write_bandit_env(tmp_path)
    results = {"rounds": 300, "n_seeds": 1, "policies": {}}
    if params is not None:
        results["params"] = params
    (tmp_path / "bandit_results.json").write_text(json.dumps(results), encoding="utf-8")
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    cfg = config.get_config()

    at = _run()
    assert not at.exception
    assert at.slider(key="bandit_epsilon").value == pytest.approx(eps if eps is not None else cfg.bandit_epsilon)
    assert at.select_slider(key="bandit_alpha").value == pytest.approx(
        alpha if alpha is not None else cfg.bandit_alpha)
    assert any(note in c.value for c in at.caption)
    at.slider(key="bandit_rounds").set_value(100)
    _button(at, "Run simulation").click()
    at.run(timeout=TIMEOUT)
    assert not at.exception


@needs_recommend
@needs_bandit
def test_bandit_lab_on_synthetic_env(tmp_path, monkeypatch):
    """Bandit Lab runs on a tiny synthetic simulator (models dir redirected to tmp_path)."""
    import cinematch.config as config

    _write_bandit_env(tmp_path)
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)

    at = _run()
    assert not at.exception
    assert any("Models not found" in w.value for w in at.warning)   # no recommender models in tmp_path
    cfg = config.get_config()                                        # no bandit_results.json -> config
    assert at.slider(key="bandit_epsilon").value == pytest.approx(cfg.bandit_epsilon)
    assert any("Defaults come from the config" in c.value for c in at.caption)
    at.slider(key="bandit_rounds").set_value(300)
    _button(at, "Run simulation").click()
    at.run(timeout=TIMEOUT)
    assert not at.exception
    results = at.session_state["bandit_results"]
    assert set(results) == {"Random", "Epsilon-greedy", "LinUCB"}
    assert results["LinUCB"]["cum_regret"][-1] < results["Random"]["cum_regret"][-1]


def _write_anomaly(path: Path) -> None:
    """A synthetic detector over anomaly.FEATURES (stored as features): standard-normal density, epsilon -12."""
    import numpy as np

    from cinematch import anomaly

    n = len(anomaly.FEATURES)
    np.savez(path, mu=np.zeros(n), var=np.ones(n), eps=-12.0, feat_mean=np.zeros(n), feat_std=np.ones(n),
             features=np.array(anomaly.FEATURES))


@needs_recommend
def test_anomaly_demo_scores_and_flags_push_bot(tmp_path, monkeypatch):
    """The bot demo builds one input per stored feature name and flags the push-bot preset."""
    import cinematch.config as config
    from cinematch import anomaly

    _write_anomaly(tmp_path / "anomaly.npz")
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    at = _run()
    assert not at.exception
    assert all(at.number_input(key=f"anom_Average real user_{f}") is not None for f in anomaly.FEATURES)
    assert sum(str(n.key).startswith("anom_") for n in at.number_input) == len(anomaly.FEATURES)
    assert any(m.label == "Log-density score" for m in at.metric)
    assert any("normal user" in s.value for s in at.success)

    at.radio(key="anom_preset").set_value("Push bot").run(timeout=TIMEOUT)
    assert not at.exception
    assert any("Flagged" in e.value for e in at.error)


@needs_recommend
def test_anomaly_demo_uses_stored_feature_names(tmp_path, monkeypatch):
    """The demo follows the names AND order stored in anomaly.npz, not anomaly.FEATURES.

    Stores the 6 FEATURES_PLUS names (like an older real artifact) in reverse order, with a distinct
    mean per stored column: each input must show its own column's mean, and the average user must
    score at the density peak (z = 0 everywhere), which only holds if values are fed in stored order.
    """
    import numpy as np

    import cinematch.config as config
    from cinematch import anomaly

    names = list(reversed(anomaly.FEATURES_PLUS))
    n = len(names)
    assert n != len(anomaly.FEATURES)
    feat_mean = np.arange(1.0, n + 1.0) * 2.0                      # 2, 4, ..., 12: one per stored column
    np.savez(tmp_path / "anomaly.npz", mu=np.zeros(n), var=np.ones(n), eps=-12.0, feat_mean=feat_mean,
             feat_std=np.ones(n), features=np.array(names, dtype="U32"))
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    at = _run()
    assert not at.exception
    anom_inputs = [x for x in at.number_input if str(x.key).startswith("anom_")]
    assert len(anom_inputs) == n
    for name, mean in zip(names, feat_mean):
        assert at.number_input(key=f"anom_Average real user_{name}").value == pytest.approx(mean)
    peak = -0.5 * n * np.log(2 * np.pi)                            # log N(0; 0, 1) summed over n features
    score = next(m for m in at.metric if m.label == "Log-density score")
    assert float(score.value) == pytest.approx(peak, abs=0.01)
    assert any("normal user" in s.value for s in at.success)


@needs_recommend
def test_how_it_works_fast_metrics_with_nulls(tmp_path, monkeypatch):
    """A FAST metrics.json with null bandit/dqn sections renders the banner and skips those sections."""
    import json

    import cinematch.config as config

    reports = tmp_path / "reports"
    reports.mkdir()
    metrics = {"generated_at": "2026-01-01T00:00:00+00:00", "dataset": "MovieLens-1M", "split": "test split",
               "fast": True, "data": {"n_users": 10, "n_items": 5},
               "rating_models": {"cf_mf": {"rmse": 0.9, "precision_at_10": 0.1, "recall_at_10": 0.1,
                                           "ndcg_at_10": 0.1},
                                 "bias_baseline": {"rmse": 0.95}},
               "anomaly": None, "clustering": {"k": 4, "silhouette": 0.05}, "bandit": None, "dqn": None}
    (reports / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    monkeypatch.setattr(config, "REPORTS_DIR", reports)
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    at = _run()
    assert not at.exception
    assert any("FAST MODE" in w.value for w in at.warning)
    assert any(m.label == "MF test RMSE" for m in at.metric)
    assert any("not shipped" in c.value for c in at.caption)
    assert not any("**Contextual bandit** (" in m.value for m in at.markdown)   # null section hidden
    assert not any("DQN session ordering" in m.value for m in at.markdown)      # dqn null -> no table
    assert not any("DQN (stretch goal)" in m.value for m in at.markdown)
    assert not any("are tuned on simulated users" in m.value for m in at.markdown)   # no tuning recorded
    assert any("coarse segmentation" in m.value for m in at.markdown)


@needs_recommend
def test_how_it_works_without_metrics(tmp_path, monkeypatch):
    import cinematch.config as config

    monkeypatch.setattr(config, "REPORTS_DIR", tmp_path)
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    at = _run()
    assert not at.exception
    assert any("metrics.json not found" in i.value for i in at.info)
    assert any("bandit simulator has not been built" in i.value for i in at.info)
    assert any("Anomaly model not found" in i.value for i in at.info)


@needs_recommend
def test_retrained_artifact_is_picked_up_without_restart(tmp_path, monkeypatch):
    """Cached loaders key on file mtimes: a model written while the app runs is used on the next rerun."""
    import os

    import cinematch.config as config

    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    at = _run()
    assert any("Anomaly model not found" in i.value for i in at.info)

    _write_anomaly(tmp_path / "anomaly.npz")
    os.utime(tmp_path / "anomaly.npz", (1e9, 1e9))
    at.run(timeout=TIMEOUT)                         # same server process, caches not cleared
    assert not at.exception
    assert not any("Anomaly model not found" in i.value for i in at.info)
    assert any(m.label == "Log-density score" for m in at.metric)


@needs_recommend
def test_how_it_works_dqn_table_ablation_and_tuned_bandit(tmp_path, monkeypatch):
    """Round-3 metrics keys: DQN dict (18.5), anomaly ablation_plus_item_dev, bandit params/no-signal share."""
    import json

    import cinematch.config as config

    reports = tmp_path / "reports"
    reports.mkdir()
    ablation = {"epsilon": -9.0, "val_f1": 0.7, "test_precision": 0.8, "test_recall": 0.6, "test_f1": 0.69,
                "per_bot_type_recall": {"push": 1.0}}
    metrics = {"generated_at": "2026-01-01T00:00:00+00:00", "dataset": "MovieLens-1M", "split": "test split",
               "fast": False, "data": {"n_users": 10, "n_items": 5}, "rating_models": {},
               "anomaly": {"test_precision": 0.9, "test_recall": 0.6, "test_f1": 0.72,
                           "per_bot_type_recall": {"push": 1.0}, "ablation_plus_item_dev": ablation},
               "clustering": None,
               "bandit": {"rounds": 3000, "n_seeds": 5,
                          "policies": {"random": {"mean_reward": 0.02, "cum_regret": 50.0, "cum_regret_std": 2.0},
                                       "linucb": {"mean_reward": 0.03, "cum_regret": 30.0, "cum_regret_std": 2.0}},
                          "params": {"epsilon": 0.05, "alpha": 0.03, "ridge": 1.0, "tuned_on_users": 1000},
                          "frac_users_no_signal": 0.619},
               "dqn": {"dqn": 0.61, "greedy_cf": 0.58, "random": 0.42, "diversity_heuristic": 0.63,
                       "n_train_users": 150, "n_eval_users": 151, "episodes": 3000, "session_len": 5}}
    (reports / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    monkeypatch.setattr(config, "REPORTS_DIR", reports)
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    at = _run()
    assert not at.exception
    assert any("DQN session ordering" in m.value for m in at.markdown)
    assert any("DQN (stretch goal)" in m.value for m in at.markdown)
    assert not any("not shipped" in c.value for c in at.caption)
    frames = [d.value for d in at.dataframe]
    dqn_frames = [f for f in frames if "Expected clicks" in f.columns]
    assert len(dqn_frames) == 1
    assert list(dqn_frames[0]["Model"]) == ["DQN (greedy policy)", "CF order", "Diversity heuristic", "Random order"]
    assert dqn_frames[0]["Expected clicks"].tolist() == [0.61, 0.58, 0.63, 0.42]
    anomaly_frames = [f for f in frames if "F1" in f.columns]
    assert anomaly_frames and "+ item-deviation feature (ablation)" in list(anomaly_frames[0]["Model"])
    captions = " ".join(c.value for c in at.caption)
    assert "tuned on 1,000 disjoint simulated users" in captions and "62%" in captions
