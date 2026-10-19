"""Bandit simulator + policies on synthetic environments (no Recommender, no real data)."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from cinematch import bandit
from cinematch.config import Config

SRC = Path(__file__).resolve().parents[1] / "src"


def _cfg(**kw) -> Config:
    cfg = Config(fast=True)
    cfg.bandit_rounds, cfg.bandit_seeds = 2000, (0, 1)
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


def synthetic_env(arm_means=(0.1, 0.2, 0.6, 0.3), n_users=50, n_clusters=3, seed=0) -> dict:
    """Every user has the same per-arm click probability; contexts are random but irrelevant."""
    rng = np.random.default_rng(seed)
    n_arms = len(arm_means)
    probs = np.broadcast_to(np.asarray(arm_means, np.float32)[None, :, None],
                            (n_users, n_arms, bandit.LIST_LEN)).copy()
    labels = rng.integers(n_clusters, size=n_users)
    contexts = bandit.build_contexts(rng.normal(size=(n_users, 40)), labels, n_clusters)
    return {"user_idx": np.arange(n_users), "lists": np.zeros(probs.shape, np.int32), "probs": probs,
            "expected": probs.mean(axis=2), "contexts": contexts, "arm_names": np.array(bandit.ARMS, dtype="U20")}


def contextual_env(n_users=60) -> dict:
    """Two user groups with opposite best arms: only a context-aware policy can serve both well."""
    env = synthetic_env(n_users=n_users, n_clusters=2)
    labels = np.arange(n_users) % 2
    env["contexts"] = bandit.build_contexts(np.zeros((n_users, 40)), labels, 2)
    means = np.where(labels[:, None] == 0, [0.6, 0.1, 0.2, 0.2], [0.1, 0.6, 0.2, 0.2]).astype(np.float32)
    env["probs"] = np.repeat(means[:, :, None], bandit.LIST_LEN, axis=2)
    env["expected"] = env["probs"].mean(axis=2)
    return env


def test_import_does_not_load_torch():
    code = "import cinematch.bandit, sys; assert 'torch' not in sys.modules"
    env = {**os.environ, "PYTHONPATH": str(SRC)}
    subprocess.run([sys.executable, "-c", code], check=True, env=env)


def test_env_step_and_expected():
    env = bandit.BanditEnv(synthetic_env(), np.random.default_rng(0))
    u = env.sample_user()
    assert 0 <= u < env.n_users
    rewards = [env.step(u, 2) for _ in range(500)]
    assert set(np.round(np.array(rewards) * 10, 6)) <= set(range(11))      # {0, .1, ..., 1}
    assert np.mean(rewards) == pytest.approx(0.6, abs=0.03)
    np.testing.assert_allclose(env.expected(u), [0.1, 0.2, 0.6, 0.3], atol=1e-6)
    assert env.context(u).shape == (4 + 3,)


def test_learning_policies_beat_random():
    cfg = _cfg()
    res = bandit.run_all(synthetic_env(), cfg)
    pol = res["policies"]
    assert pol["linucb"]["cum_regret"] < 0.5 * pol["random"]["cum_regret"]
    assert pol["epsilon_greedy"]["cum_regret"] < 0.5 * pol["random"]["cum_regret"]
    assert pol["linucb"]["mean_reward"] > pol["random"]["mean_reward"]
    assert int(np.argmax(pol["linucb"]["arm_counts"])) == 2                # the clearly best arm
    assert res["oracle_mean_reward"] == pytest.approx(0.6, abs=1e-4)
    assert res["arm_mean_expected_reward"] == {"cf": 0.1, "content": 0.2, "cluster_popular": 0.6, "trending": 0.3}
    assert res["rounds"] == 2000 and res["n_seeds"] == 2
    assert res["frac_users_no_signal"] == 0.0
    assert res["params"] == {"epsilon": cfg.bandit_epsilon, "alpha": cfg.bandit_alpha, "ridge": cfg.bandit_ridge}


def test_best_fixed_arm_uses_same_sequences():
    """Hindsight best arm = highest mean expected reward; played on the policies' user sequences."""
    cfg = _cfg(bandit_rounds=500)
    res = bandit.run_all(synthetic_env(), cfg)
    fixed = res["best_fixed_arm"]
    assert fixed["arm"] == "cluster_popular" and fixed["cum_regret"] == 0.0     # best arm for every user
    assert fixed["mean_reward"] == pytest.approx(0.6, abs=0.03)
    assert len(fixed["cum_regret_curve"]) == len(res["policies"]["random"]["cum_regret_curve"])
    # contextual env: arms 0 and 1 near-tied on average; regret = sum over the SAME users as random's
    env = contextual_env()
    env["probs"] = env["probs"] * (1 + 0.001 * np.arange(60, dtype=np.float32))[:, None, None]  # unique oracle per user
    env["expected"] = env["probs"].mean(axis=2)
    res = bandit.run_all(env, cfg)
    fixed = res["best_fixed_arm"]
    assert fixed["arm"] == "content"                                       # group 1 is scaled up slightly more
    regrets = []
    for seed in cfg.bandit_seeds:
        users = bandit.run_simulation(bandit.BanditEnv(env, np.random.default_rng(seed)),
                                      bandit.RandomPolicy(4, np.random.default_rng(9)), cfg.bandit_rounds)
        sim = bandit.run_simulation(bandit.BanditEnv(env, np.random.default_rng(seed)),
                                    bandit.FixedArmPolicy(4, np.random.default_rng(9), 1), cfg.bandit_rounds)
        np.testing.assert_array_equal(sim["oracle"], users["oracle"])           # same user sequence
        assert (sim["arms"] == 1).all()
        regrets.append(sim["regret"].sum())
    assert fixed["cum_regret"] == pytest.approx(np.mean(regrets), abs=1e-3)
    assert fixed["cum_regret_std"] == pytest.approx(np.std(regrets, ddof=1), abs=1e-3)
    assert res["policies"]["linucb"]["cum_regret"] < fixed["cum_regret"]       # context beats any single arm


def test_seed_std_is_sample_std_and_no_signal_fraction():
    final = np.array([1.0, 2.0, 4.0])
    assert bandit._seed_std(final) == pytest.approx(np.std(final, ddof=1))
    assert bandit._seed_std(np.array([3.0])) == 0.0
    expected = np.array([[0.01] * 4, [0.01, 0.2, 0.01, 0.01], [0.0] * 4, [0.1, 0.1, 0.1, 0.3]])
    assert bandit.frac_users_no_signal(expected) == pytest.approx(0.5)


def test_linucb_matches_explicit_inverse():
    rng = np.random.default_rng(3)
    dim, n_arms, alpha, ridge = 6, 3, 0.7, 2.0
    policy = bandit.LinUCB(n_arms, dim, np.random.default_rng(0), alpha=alpha, ridge=ridge)
    A = np.repeat(ridge * np.eye(dim)[None], n_arms, axis=0)
    b = np.zeros((n_arms, dim))
    for _ in range(50):
        x, arm, r = rng.normal(size=dim), int(rng.integers(n_arms)), float(rng.random())
        policy.update(x, arm, r)
        A[arm] += np.outer(x, x)
        b[arm] += r * x
    A_inv = np.linalg.inv(A)
    np.testing.assert_allclose(policy.A_inv, A_inv, rtol=1e-8, atol=1e-12)
    x = rng.normal(size=dim)
    theta = np.einsum("kij,kj->ki", A_inv, b)
    want = theta @ x + alpha * np.sqrt(np.einsum("i,kij,j->k", x, A_inv, x))
    np.testing.assert_allclose(policy.scores(x), want, rtol=1e-8)


def test_tune_alpha_and_sample_users():
    res = bandit.tune_alpha(synthetic_env(), _cfg(bandit_rounds=200), alphas=(1.0, 0.1), seeds=(100, 101))
    assert set(res) == {1.0, 0.1} and all(v["cum_regret"] >= 0 and v["cum_regret_std"] >= 0 for v in res.values())
    users = np.arange(0, 40, 2)
    reported = bandit.sample_users(users, 8, seed=0)
    np.testing.assert_array_equal(reported, np.sort(np.random.default_rng(0).choice(users, 8, replace=False)))
    np.testing.assert_array_equal(bandit.sample_users(users, 100, seed=0), users)      # fewer than n: all


def test_linucb_uses_context():
    res = bandit.run_all(contextual_env(), _cfg(bandit_rounds=1500))
    pol = res["policies"]
    assert pol["linucb"]["cum_regret"] < 0.5 * pol["epsilon_greedy"]["cum_regret"]


def test_regret_nonnegative_and_monotone():
    env = bandit.BanditEnv(synthetic_env(), np.random.default_rng(1))
    sim = bandit.run_simulation(env, bandit.LinUCB(4, env.dim, np.random.default_rng(2)), 300)
    assert len(sim["reward"]) == len(sim["regret"]) == len(sim["arms"]) == 300
    assert (sim["regret"] >= 0).all()
    assert np.all(np.diff(np.cumsum(sim["regret"])) >= 0)
    res = bandit.run_all(synthetic_env(), _cfg(bandit_rounds=305))
    for p in res["policies"].values():
        curve = np.array(p["cum_regret_curve"])
        assert len(curve) == 31 and np.all(np.diff(curve) >= 0)           # 30 points + final round
        assert np.all(np.diff(p["cum_reward_curve"]) >= 0)
        assert sum(p["arm_counts"]) == pytest.approx(305)


def test_seed_determinism():
    cfg = _cfg(bandit_rounds=400)
    assert bandit.run_all(synthetic_env(), cfg) == bandit.run_all(synthetic_env(), cfg)
    other = bandit.run_all(synthetic_env(), _cfg(bandit_rounds=400, bandit_seeds=(5, 6)))
    assert other != bandit.run_all(synthetic_env(), cfg)


class StubRecommender:
    """recommend_strategy(u, arm) -> items [base + u, ...]: arm a lists items 10a..10a+9 (shifted)."""

    def recommend_strategy(self, u, strategy, n=10):
        a = bandit.ARMS.index(strategy)
        if strategy == "trending":
            return np.arange(30, 30 + n - 2)                                   # short list -> padding
        return np.arange(10 * a, 10 * a + n)


def test_precompute_click_model_and_contexts():
    cfg = _cfg()
    users = np.array([0, 2])
    holdout = {0: {0: 5.0, 1: 3.0, 15: 1.0}, 1: {0: 5.0}, 2: {12: 4.0}}
    feat = np.random.default_rng(0).normal(size=(3, 40))
    env = bandit.precompute_env(cfg, StubRecommender(), users, holdout, feat, np.array([1, 0, 2]), 3)
    assert env["lists"].shape == (2, 4, 10) and env["probs"].dtype == np.float32
    p = env["probs"]
    assert p[0, 0, 0] == pytest.approx(0.75) and p[0, 0, 1] == pytest.approx(0.15)
    assert p[0, 0, 2] == pytest.approx(cfg.click_unrated)
    assert p[0, 1, 5] == pytest.approx(0.02)                                   # item 15 rated 1
    assert p[1, 1, 2] == pytest.approx(0.45)                                   # user 2, item 12 rated 4
    assert (env["lists"][:, 3, 8:] == -1).all() and (p[:, 3, 8:] == 0).all()   # padded slots
    np.testing.assert_allclose(env["expected"], p.mean(axis=2), rtol=1e-6)
    ctx = env["contexts"]
    assert ctx.shape == (2, 4 + 3)
    np.testing.assert_array_equal(ctx[:, 0], 1.0)
    np.testing.assert_allclose(ctx[:, 1:4].mean(axis=0), 0.0, atol=1e-6)        # standardised
    np.testing.assert_array_equal(ctx[:, 4:], [[0, 1, 0], [0, 0, 1]])          # users 0, 2 -> clusters 1, 2


def test_train_and_eval_step_roundtrip(tmp_path):
    models, figs = tmp_path / "models", tmp_path / "figs"
    models.mkdir()
    n_users = 30
    rng = np.random.default_rng(0)
    np.save(models / "eval_users.npy", np.arange(0, n_users, 2))
    np.savez(models / "two_tower.npz", user_feat_raw=rng.normal(size=(n_users, 40)).astype(np.float32))
    np.savez(models / "clusters.npz", labels=np.arange(n_users) % 3, centroids=np.zeros((3, 4)))
    holdout = {u: {10 * (u % 4): 5.0, 1: 4.0} for u in range(n_users)}
    cfg = _cfg(bandit_users=10, bandit_rounds=120)
    ctx = {"models_dir": models, "figures_dir": figs, "processed_dir": tmp_path / "nope",
           "recommender": StubRecommender(), "data": {"holdout_test": holdout}}
    out = bandit.train_step(cfg, ctx)
    env = bandit.load_env(models)
    assert env["user_idx"].shape == (10,) and set(env["user_idx"]) <= set(range(0, n_users, 2))
    assert list(env["arm_names"]) == bandit.ARMS and env["contexts"].shape == (10, 7)
    results = json.loads((models / "bandit_results.json").read_text(encoding="utf-8"))
    assert set(results) == {"policies", "oracle_mean_reward", "arm_mean_expected_reward", "best_fixed_arm", "rounds",
                            "n_seeds", "frac_users_no_signal", "params", "tuning", "tuning_std"}
    assert set(results["params"]) == {"epsilon", "alpha", "ridge", "tuned_on_users"}
    assert results["params"]["tuned_on_users"] == 15                      # odd users: not in eval_users.npy
    assert results["params"]["alpha"] in cfg.bandit_alpha_grid
    assert results["params"]["epsilon"] in cfg.bandit_epsilon_grid
    assert list(results["tuning"]["alpha"]) == [str(a) for a in cfg.bandit_alpha_grid]
    assert list(results["tuning"]["epsilon"]) == [str(e) for e in cfg.bandit_epsilon_grid]
    assert {k: list(v) for k, v in results["tuning_std"].items()} == {k: list(v) for k, v in results["tuning"].items()}
    assert set(results["policies"]["linucb"]) == {"mean_reward", "cum_regret", "cum_regret_std", "arm_counts",
                                                  "cum_reward_curve", "cum_regret_curve", "cum_regret_curve_std"}
    metrics = bandit.eval_step(cfg, ctx)
    assert metrics == out
    assert set(metrics["policies"]) == {"random", "epsilon_greedy", "linucb"}
    assert "cum_regret_curve" not in metrics["policies"]["random"]
    assert metrics["params"] == results["params"] and metrics["tuning"] == results["tuning"]
    assert metrics["tuning_std"] == results["tuning_std"]
    assert "frac_users_no_signal" in metrics
    assert set(metrics["best_fixed_arm"]) == {"arm", "mean_reward", "cum_regret", "cum_regret_std"}
    assert "cum_regret_curve" in results["best_fixed_arm"]
    assert (figs / "bandit_cumulative_reward.png").exists() and (figs / "bandit_regret.png").exists()


def test_train_step_default_holdout_uses_test_split_only(tmp_path, monkeypatch):
    """Without ctx["data"], clicks come from load_processed()["ratings"] TEST rows; val rows are ignored."""
    import pandas as pd
    from cinematch import data

    models = tmp_path / "models"
    models.mkdir()
    n_users = 6
    np.save(models / "eval_users.npy", np.arange(n_users))
    np.savez(models / "two_tower.npz", user_feat_raw=np.random.default_rng(0).normal(size=(n_users, 40)))
    np.savez(models / "clusters.npz", labels=np.arange(n_users) % 2, centroids=np.zeros((2, 4)))
    rows = [(u, 0, 5.0, "val") for u in range(n_users)] + [(u, 1, 4.0, "test") for u in range(n_users)]
    rows += [(u, 2, 5.0, "train") for u in range(n_users)]
    ratings = pd.DataFrame(rows, columns=["u", "i", "rating", "split"])
    monkeypatch.setattr(data, "load_processed", lambda processed_dir=None: {"ratings": ratings})
    cfg = _cfg(bandit_users=n_users, bandit_rounds=50)
    bandit.train_step(cfg, {"models_dir": models, "figures_dir": tmp_path / "figs",
                            "processed_dir": tmp_path / "nope", "recommender": StubRecommender()})
    env = bandit.load_env(models)
    cf = bandit.ARMS.index("cf")
    assert (env["lists"][:, cf, :2] == [0, 1]).all()                       # StubRecommender cf list = 0..9
    np.testing.assert_allclose(env["probs"][:, cf, 0], cfg.click_unrated)   # val-only item 0
    np.testing.assert_allclose(env["probs"][:, cf, 1], cfg.click_probs[4])  # test-only item 1
    np.testing.assert_allclose(env["probs"][:, cf, 2], cfg.click_unrated)   # train-only item 2


def test_eval_step_missing_returns_none(tmp_path):
    assert bandit.eval_step(_cfg(), {"models_dir": tmp_path, "figures_dir": tmp_path}) is None
    with pytest.raises(FileNotFoundError):
        bandit.load_env(tmp_path)


def test_tune_params_picks_lowest_regret():
    """One clearly best arm for everyone: a small alpha / epsilon wastes less on exploration.

    (alpha=0.01 is too greedy and can lock onto a bad arm on one seed, so 0.1 wins the grid.)
    """
    cfg = _cfg(bandit_rounds=1500, bandit_tune_seeds=(100, 101))
    params, tuning, tuning_std = bandit.tune_params(synthetic_env(), cfg)
    assert set(tuning) == set(tuning_std) == {"alpha", "epsilon"}
    for key in ("alpha", "epsilon"):
        assert params[key] == float(min(tuning[key], key=tuning[key].get))       # argmin of the table
        assert list(tuning_std[key]) == list(tuning[key]) and min(tuning_std[key].values()) >= 0
    assert params["alpha"] < 1.0 and tuning["alpha"][str(params["alpha"])] < 0.2 * tuning["alpha"]["1.0"]
    assert params["epsilon"] < 0.2 and tuning["epsilon"][str(params["epsilon"])] < tuning["epsilon"]["0.2"]
    assert bandit.tune_epsilon(synthetic_env(), cfg, (0.2,), (100,))[0.2]["cum_regret_std"] == 0.0


def test_best_picks_lowest_regret_and_earlier_grid_value_on_tie():
    assert bandit._best({1.0: {"cum_regret": 5.0}, 0.1: {"cum_regret": 5.0}, 0.03: {"cum_regret": 6.0}}) == 1.0
    assert bandit._best({1.0: {"cum_regret": 5.0}, 0.1: {"cum_regret": 4.0}}) == 0.1


def test_tuning_user_pool_excludes_eval_flagged_and_no_test():
    holdout = {0: {1: 4.0}, 1: {2: 5.0}, 2: {}, 3: {1: 3.0}, 4: {7: 1.0}, 5: {3: 2.0}}
    pool = bandit.tuning_user_pool(holdout, eval_users=np.array([0, 4]), flagged=np.array([3]))
    np.testing.assert_array_equal(pool, [1, 5])                            # 2 has no test row


class RecordingRecommender(StubRecommender):
    def __init__(self):
        self.users = set()

    def recommend_strategy(self, u, strategy, n=10):
        self.users.add(int(u))
        return super().recommend_strategy(u, strategy, n)


def test_train_step_tunes_on_disjoint_users(tmp_path, monkeypatch):
    """Tuning env = non-eval, non-flagged users with a test row; its picks drive the reported run."""
    models = tmp_path / "models"
    models.mkdir()
    n_users = 40
    eval_users = np.arange(0, n_users, 2)
    flagged = np.array([1, 3, 5])
    np.save(models / "eval_users.npy", eval_users)
    np.savez(models / "two_tower.npz", user_feat_raw=np.random.default_rng(0).normal(size=(n_users, 40)))
    np.savez(models / "clusters.npz", labels=np.arange(n_users) % 2, centroids=np.zeros((2, 4)))
    holdout = {u: {10 * (u % 4): 5.0} for u in range(n_users) if u != 7}     # user 7: no test row
    index_maps = {"user_ids": list(range(1000, 1000 + n_users)), "flagged_user_ids": [1001, 1003, 1005]}
    seen = {}

    def fake_tune(tune_env, cfg):
        seen["users"] = tune_env["user_idx"].copy()
        return ({"epsilon": 0.05, "alpha": 0.03}, {"alpha": {"0.03": 1.0}, "epsilon": {"0.05": 2.0}},
                {"alpha": {"0.03": 0.5}, "epsilon": {"0.05": 0.7}})

    monkeypatch.setattr(bandit, "tune_params", fake_tune)
    rec = RecordingRecommender()
    cfg = _cfg(bandit_users=8, bandit_rounds=60, bandit_tune_users=10)
    out = bandit.train_step(cfg, {"models_dir": models, "figures_dir": tmp_path / "figs",
                                  "processed_dir": tmp_path / "nope", "recommender": rec,
                                  "data": {"holdout_test": holdout, "index_maps": index_maps}})
    tune = set(seen["users"].tolist())
    assert len(tune) == 10
    assert not tune & set(eval_users.tolist()) and not tune & set(flagged.tolist()) and 7 not in tune
    reported = set(bandit.load_env(models)["user_idx"].tolist())
    assert reported <= set(eval_users.tolist()) and not reported & tune
    assert rec.users == reported | tune
    assert out["params"] == {"epsilon": 0.05, "alpha": 0.03, "ridge": cfg.bandit_ridge, "tuned_on_users": 10}
    assert out["tuning"] == {"alpha": {"0.03": 1.0}, "epsilon": {"0.05": 2.0}}
    assert out["tuning_std"] == {"alpha": {"0.03": 0.5}, "epsilon": {"0.05": 0.7}}
    assert cfg.bandit_alpha == 1.0 and cfg.bandit_epsilon == 0.1            # caller's cfg is not mutated


def test_fast_config_tuning_fields(monkeypatch):
    monkeypatch.delenv("CINEMATCH_FAST", raising=False)          # CINEMATCH_FAST=1 would force fast mode
    full, fast = Config(fast=False), Config(fast=True)
    assert full.bandit_alpha_grid == (1.0, 0.3, 0.1, 0.03, 0.01)
    assert full.bandit_epsilon_grid == (0.2, 0.1, 0.05, 0.02)
    assert (full.bandit_tune_users, full.bandit_tune_seeds) == (1000, (100, 101))
    assert (fast.bandit_tune_users, fast.bandit_tune_seeds) == (100, (100,))
    assert not set(full.bandit_tune_seeds) & set(full.bandit_seeds)


def test_arm_labels_match_readme_fill():
    """The app uses bandit.ARM_LABELS, the README uses readme_fill's numpy-free copy: both must agree."""
    from cinematch import readme_fill

    assert set(bandit.ARM_LABELS) == set(bandit.ARMS)
    assert bandit.ARM_LABELS == readme_fill.ARM_LABELS
    assert bandit.POLICY_LABELS == readme_fill.POLICY_LABELS
