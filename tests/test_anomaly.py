"""Tests for cinematch.anomaly (self-contained synthetic data, all outputs in tmp_path)."""
import json
import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from cinematch import anomaly
from cinematch.config import Config

DAY = 86400


def _synthetic_raw(n_users=400, n_movies=300, seed=0):
    """Realistic-ish real users: moderate ratings, spread over weeks/months, no bots."""
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(1, n_users + 1):
        n = int(rng.integers(20, 150))
        mids = rng.choice(np.arange(1, n_movies + 1), size=n, replace=False)
        r = np.clip(np.round(rng.normal(3.6, 0.35) + rng.normal(0, 0.9, n)), 1, 5)
        start = int(rng.integers(956_000_000, 1_000_000_000))
        span = int(rng.integers(30, 900)) * DAY
        ts = start + np.sort(rng.integers(0, span, n))
        rows.append(pd.DataFrame({"user_id": u, "movie_id": mids, "rating": r, "timestamp": ts}))
    ratings = pd.concat(rows, ignore_index=True).astype(
        {"user_id": np.int64, "movie_id": np.int64, "rating": np.float32, "timestamp": np.int64})
    movies = pd.DataFrame({"movie_id": np.arange(1, n_movies + 1),
                           "title": [f"Movie {i} (1995)" for i in range(1, n_movies + 1)],
                           "genres": "Drama"})
    return ratings, movies


def _cfg(**kw):
    cfg = Config(fast=True)
    cfg.anomaly_bots_per_type = 20
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


def test_user_features_known_values():
    ratings = pd.DataFrame({
        "user_id": [1, 1, 1, 2, 2, 3],
        "movie_id": [10, 11, 12, 10, 11, 10],
        "rating": np.array([5, 1, 3, 4, 4, 2], dtype=np.float32),
        "timestamp": [0, DAY, 2 * DAY, 500, 500, 7],
    })
    f = anomaly.user_features(ratings, with_item_dev=True)
    assert list(f.columns) == anomaly.FEATURES_PLUS
    assert list(f.index) == [1, 2, 3]
    f5 = anomaly.user_features(ratings)
    assert list(f5.columns) == anomaly.FEATURES and len(anomaly.FEATURES) == 5
    pd.testing.assert_frame_equal(f5, f[anomaly.FEATURES])
    expected = {
        # item means: movie 10 -> 11/3, movie 11 -> 2.5, movie 12 -> 3
        1: [math.log(4), 3.0, 8 / 3, 2 / 3, math.log(2), 17 / 18],   # span 2 days -> 3 / 3 per day
        2: [math.log(3), 4.0, 0.0, 0.0, math.log(3), 11 / 12],       # one session -> 2 / 1
        3: [math.log(2), 2.0, 0.0, 0.0, math.log(2), 5 / 3],
    }
    for uid, vals in expected.items():
        np.testing.assert_allclose(f.loc[uid].to_numpy(), vals, rtol=1e-6)


def test_mean_abs_item_dev_hand_computed():
    ratings = pd.DataFrame({
        "user_id": [1, 1, 2, 2, 3],
        "movie_id": [7, 8, 7, 8, 7],
        "rating": np.array([4, 2, 2, 4, 3], dtype=np.float32),
        "timestamp": [0, 10, 0, 10, 0],
    })
    # item means over THIS frame: movie 7 -> 3, movie 8 -> 3
    f = anomaly.user_features(ratings, with_item_dev=True)
    assert anomaly.FEATURES_PLUS == anomaly.FEATURES + ["mean_abs_item_dev"]
    np.testing.assert_allclose(f["mean_abs_item_dev"].to_numpy(), [1.0, 1.0, 0.0])
    # A bot rating movie 7 with 5 shifts the item mean (5 -> 14/4 = 3.5): item means are
    # computed on the frame passed in, bots included.
    with_bot = pd.concat([ratings, pd.DataFrame({"user_id": [100000], "movie_id": [7],
                                                 "rating": np.array([5], dtype=np.float32),
                                                 "timestamp": [0]})], ignore_index=True)
    fb = anomaly.user_features(with_bot, with_item_dev=True)["mean_abs_item_dev"]
    np.testing.assert_allclose(fb.loc[[1, 2, 3, 100000]].to_numpy(),
                               [(0.5 + 1.0) / 2, (1.5 + 1.0) / 2, 0.5, 1.5])


def test_inject_bots_counts_and_types():
    ratings, movies = _synthetic_raw(n_users=50)
    out, bots = anomaly.inject_bots(ratings, movies, n_per_type=6, seed=1)
    assert len(bots) == 30
    assert bots["bot_type"].value_counts().to_dict() == {t: 6 for t in anomaly.BOT_TYPES}
    assert bots["user_id"].min() == 100000 and bots["user_id"].is_unique
    bot_rows = out[out["user_id"] >= 100000]
    assert set(bot_rows["user_id"]) == set(bots["user_id"])
    assert len(out) == len(ratings) + len(bot_rows)
    assert bot_rows["rating"].between(1, 5).all()
    assert not bot_rows.duplicated(["user_id", "movie_id"]).any()
    assert out["rating"].dtype == ratings["rating"].dtype
    sizes = bot_rows.groupby("user_id").size()
    assert sizes.min() >= 30
    spans = bot_rows.groupby("user_id")["timestamp"].agg(lambda s: s.max() - s.min())
    type_of = bots.set_index("user_id")["bot_type"]
    assert (spans[type_of[type_of == "push"].index] <= 3600).all()
    assert (spans[type_of[type_of == "camouflage_slow"].index] > 3 * DAY).all()
    # deterministic
    out2, _ = anomaly.inject_bots(ratings, movies, n_per_type=6, seed=1)
    pd.testing.assert_frame_equal(out, out2)


def test_log_density_matches_scipy():
    rng = np.random.default_rng(0)
    X = rng.normal(2.0, 3.0, size=(500, 3))
    mu, var = anomaly.fit_gaussian(X)
    np.testing.assert_allclose(mu, X.mean(0))
    np.testing.assert_allclose(var, X.var(0))
    expected = norm.logpdf(X, loc=mu, scale=np.sqrt(var)).sum(axis=1)
    np.testing.assert_allclose(anomaly.log_density(X, mu, var), expected, rtol=1e-10)
    _, var_const = anomaly.fit_gaussian(np.ones((10, 2)))
    assert np.all(var_const == 1e-6)


def test_evaluate_threshold_hand_values():
    scores = np.array([-10.0, -8.0, -1.0, -0.5, -0.2])
    y = np.array([1, 0, 1, 0, 0])
    res = anomaly.evaluate_threshold(scores, y, eps=-0.8)   # flags -10, -8, -1
    assert res["n_flagged"] == 3
    assert res["precision"] == pytest.approx(2 / 3)
    assert res["recall"] == pytest.approx(1.0)
    assert res["f1"] == pytest.approx(0.8)
    assert anomaly.evaluate_threshold(scores, y, eps=-100)["f1"] == 0.0


def test_select_epsilon_separable_f1():
    rng = np.random.default_rng(0)
    X_norm = rng.normal(0, 1, size=(1000, 4))
    X_val = np.vstack([rng.normal(0, 1, size=(300, 4)), rng.normal(5, 1, size=(30, 4))])
    y_val = np.r_[np.zeros(300), np.ones(30)]
    mu, var = anomaly.fit_gaussian(X_norm)
    eps, f1 = anomaly.select_epsilon(anomaly.log_density(X_val, mu, var), y_val, 200)
    assert f1 >= 0.8
    assert anomaly.evaluate_threshold(anomaly.log_density(X_val, mu, var), y_val, eps)["f1"] == pytest.approx(f1)


def test_select_epsilon_tie_picks_fewer_flags():
    # Grid = {-4,-3,-2,-1,0}. eps=-3 (P=1, R=0.5) and eps=0 (P=0.5, R=1) both give F1 = 2/3
    # exactly; the tie must go to the smaller epsilon (fewer flagged users).
    scores = np.array([-4.0, -3.0, -2.0, -1.0, 0.0])
    y = np.array([1, 0, 0, 1, 0])
    assert anomaly.evaluate_threshold(scores, y, 0.0)["f1"] == pytest.approx(2 / 3)
    eps, f1 = anomaly.select_epsilon(scores, y, n_grid=5)
    assert eps == -3.0 and f1 == pytest.approx(2 / 3)


def test_cap_rule():
    real = np.linspace(-10, 0, 1000)
    eps, capped = anomaly.cap_epsilon(-5.0, real)          # would flag ~50%
    assert capped and np.mean(real < eps) <= 0.03
    eps2, capped2 = anomaly.cap_epsilon(-9.9, real)        # flags ~1%
    assert not capped2 and eps2 == -9.9
    # ML-1M-sized population: an interpolated quantile would flag 182 > 0.03 * 6040 users.
    real = np.random.default_rng(0).normal(size=6040)
    eps3, capped3 = anomaly.cap_epsilon(np.inf, real)
    assert capped3
    assert np.sum(real < eps3) <= 0.03 * len(real)
    assert np.sum(real < eps3) == int(np.floor(0.03 * len(real)))


def test_train_step_capped_epsilon(tmp_path, monkeypatch):
    # Force a terrible val epsilon (flags everyone) so the 3% safety cap must fire.
    monkeypatch.setattr(anomaly, "select_epsilon", lambda s, y, n_grid=200: (float(np.max(s)) + 1.0, 0.0))
    ratings, movies = _synthetic_raw()
    ctx = {"raw_ratings": ratings, "raw_movies": movies,
           "models_dir": tmp_path / "models", "figures_dir": tmp_path / "figs"}
    report = anomaly.train_step(_cfg(), ctx)
    n_real = ratings["user_id"].nunique()
    flagged = json.loads((tmp_path / "models" / "flagged_users.json").read_text())
    assert report["epsilon_capped"] is True
    assert report["epsilon"] < report["epsilon_val"]
    assert 0 < len(flagged) <= 0.03 * n_real
    assert report["n_real_flagged_all"] == len(flagged)
    assert ctx["flagged_user_ids"] == flagged
    with np.load(tmp_path / "models" / "anomaly.npz") as f:
        eps, eps_val, scores_real = float(f["eps"]), float(f["eps_val"]), f["scores_real"]
        scores_test, y_test = f["scores_test"], f["y_test"]
    assert eps < eps_val
    assert int(np.sum(scores_real < eps)) == len(flagged)
    assert report["test_f1"] == pytest.approx(anomaly.evaluate_threshold(scores_test, y_test, eps)["f1"], abs=1e-4)
    assert report["epsilon"] == pytest.approx(eps, abs=1e-4)
    assert report["ablation_plus_item_dev_capped"] is True   # the monkeypatch caps both models
    metrics = anomaly.eval_step(_cfg(), ctx)
    assert metrics["epsilon_capped"] is True
    assert metrics["ablation_plus_item_dev_capped"] is True


def test_train_and_eval_step(tmp_path, monkeypatch):
    ratings, movies = _synthetic_raw()
    cfg = _cfg()
    ctx = {"raw_ratings": ratings, "raw_movies": movies,
           "models_dir": tmp_path / "models", "figures_dir": tmp_path / "figs"}
    calls, fitted = [], []
    orig_fit = anomaly._fit_detector

    def spy(feats, *args):
        calls.append(list(feats.columns))
        fitted.append(orig_fit(feats, *args))
        return fitted[-1]

    monkeypatch.setattr(anomaly, "_fit_detector", spy)
    report = anomaly.train_step(cfg, ctx)
    # Deployed model on the 5 spec features, ablation adds mean_abs_item_dev (spec 18.1).
    assert calls == [anomaly.FEATURES, anomaly.FEATURES_PLUS]
    assert report["features"] == anomaly.FEATURES and report["ablation_features"] == anomaly.FEATURES_PLUS
    assert "ablation_5_features" not in report
    abl_rep = report["ablation_plus_item_dev"]
    assert abl_rep["epsilon"] == pytest.approx(fitted[1]["eps"], abs=1e-4)
    assert abl_rep["test_f1"] == pytest.approx(fitted[1]["test"]["f1"], abs=1e-4)
    assert fitted[0]["feat_mean"].shape == (5,) and fitted[1]["feat_mean"].shape == (6,)
    assert report["epsilon"] == pytest.approx(fitted[0]["eps"], abs=1e-4)
    assert set(report["val_per_bot_type_recall"]) == set(anomaly.BOT_TYPES)
    assert set(report["ablation_plus_item_dev_val_per_bot_type_recall"]) == set(anomaly.BOT_TYPES)

    # Deployed scores come from the 5 features of the bot-injected frame, standardised with the
    # saved normal_train statistics; the flag list uses the deployed epsilon.
    with_bots, _ = anomaly.inject_bots(ratings, movies, cfg.anomaly_bots_per_type, cfg.seed)
    with np.load(tmp_path / "models" / "anomaly.npz", allow_pickle=False) as f:
        real_ids, scores_real, eps = f["real_user_ids"], f["scores_real"], float(f["eps"])
        Z = (anomaly.user_features(with_bots).loc[real_ids].to_numpy() - f["feat_mean"]) / f["feat_std"]
        np.testing.assert_allclose(anomaly.log_density(Z, f["mu"], f["var"]), scores_real, rtol=1e-10)
        assert [str(x) for x in f["features"]] == anomaly.FEATURES
        assert f["mu"].shape == (5,)
    # The ablation sees item means that include the bots (as a live platform would).
    plus_bots = anomaly.user_features(with_bots, with_item_dev=True).loc[real_ids, "mean_abs_item_dev"]
    plus_clean = anomaly.user_features(ratings, with_item_dev=True).loc[real_ids, "mean_abs_item_dev"]
    assert not np.allclose(plus_bots.to_numpy(), plus_clean.to_numpy())
    for name in ("anomaly.npz", "anomaly_report.json", "flagged_users.json"):
        assert (tmp_path / "models" / name).exists()
    flagged = json.loads((tmp_path / "models" / "flagged_users.json").read_text())
    assert ctx["flagged_user_ids"] == flagged
    assert all(u < 100000 for u in flagged)
    assert sorted(flagged) == sorted(int(u) for u in real_ids[scores_real < eps])
    assert len(flagged) <= 0.03 * ratings["user_id"].nunique()
    assert report["n_real_flagged_all"] == len(flagged)
    assert report["test_f1"] > 0.8, report
    assert report["n_bots_test"] == 50

    metrics = anomaly.eval_step(cfg, ctx)
    assert set(metrics) == {"epsilon", "epsilon_capped", "val_f1", "test_precision", "test_recall", "test_f1",
                            "n_bots_test", "n_real_flagged", "n_real_flagged_all", "per_bot_type_recall",
                            "ablation_plus_item_dev", "ablation_plus_item_dev_capped"}
    assert set(metrics["per_bot_type_recall"]) == set(anomaly.BOT_TYPES)
    abl = metrics["ablation_plus_item_dev"]
    assert set(abl) == {"epsilon", "val_f1", "test_precision", "test_recall", "test_f1", "per_bot_type_recall"}
    assert set(abl["per_bot_type_recall"]) == set(anomaly.BOT_TYPES)
    assert all(0.0 <= abl[k] <= 1.0 for k in ("val_f1", "test_precision", "test_recall", "test_f1"))
    assert abl == json.loads((tmp_path / "models" / "anomaly_report.json").read_text())["ablation_plus_item_dev"]
    for fig in ("anomaly_scores.png", "anomaly_pr_curve.png"):
        assert (tmp_path / "figs" / fig).stat().st_size > 0

    model = anomaly.load_anomaly(tmp_path / "models")
    assert model["features"] == anomaly.FEATURES and model["feat_mean"].shape == (5,)
    bot_like = np.array([np.log1p(200), 5.0, 0.0, 1.0, np.log1p(200)])
    assert anomaly.score_user(model, bot_like) < model["eps"]
    assert anomaly.score_user(model, model["feat_mean"]) > model["eps"]
    with pytest.raises(ValueError, match="6 features"):
        anomaly.score_user(model, np.r_[bot_like, 1.4])


def test_load_anomaly_requires_feature_names(tmp_path):
    """npz files without stored feature names are rejected with a clear 'retrain' message."""
    d = len(anomaly.FEATURES)
    np.savez(tmp_path / "anomaly.npz", mu=np.zeros(d), var=np.ones(d), eps=np.float64(-5.0),
             feat_mean=np.zeros(d), feat_std=np.ones(d))
    with pytest.raises(ValueError, match="retrain"):
        anomaly.load_anomaly(tmp_path)


def test_eval_step_missing_artifacts(tmp_path):
    assert anomaly.eval_step(_cfg(), {"models_dir": tmp_path, "figures_dir": tmp_path}) is None
    with pytest.raises(FileNotFoundError):
        anomaly.load_anomaly(tmp_path)
