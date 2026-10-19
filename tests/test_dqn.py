"""DQN session-ordering stretch: session math, baselines, analytic evaluator, train/eval round trip."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from cinematch import dqn
from cinematch.config import Config

SRC = Path(__file__).resolve().parents[1] / "src"
A, C, D, H = 0, 4, 7, 10          # genre columns: Action, Comedy, Drama, Horror


def _cfg(**kw) -> Config:
    cfg = Config(fast=True)
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


def _genres(sets) -> np.ndarray:
    """[1, 8, 18] multi-hot from a list of 8 genre-index lists."""
    g = np.zeros((1, dqn.N_CAND, dqn.N_GENRES), np.float32)
    for j, cols in enumerate(sets):
        g[0, j, cols] = 1.0
    return g


def hand_user():
    p = np.array([[0.5, 0.4, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0]])
    g = _genres([[A], [A, C], [D], [C], [A], [H], [H], [H]])
    return p, g, dqn.genre_overlap(g)


def test_session_dynamics_hand_example():
    p, g, share = hand_user()
    d, f = 0.85, 0.8
    # slot 0 (A) -> 1 (A,C shares A) -> 2 (D, new) -> 3 (C shares with 1) -> 4 (A shares with 0 and 1)
    expected = 0.5 + 0.4 * 0.85 * 0.8 + 0.3 * 0.85 ** 2 + 0.2 * 0.85 ** 3 * 0.8 + 0.1 * 0.85 ** 4 * 0.8 ** 2
    got = dqn.order_expected_reward(p, share, np.array([[0, 1, 2, 3, 4]]), d, f)
    assert got.shape == (1,)
    assert got[0] == pytest.approx(expected, abs=1e-12)
    # per-step probabilities after showing slots 0 and 1 at step 2
    used = np.array([[True, True] + [False] * 6])
    probs = dqn.step_click_probs(p, share, used, 2, d, f)[0]
    assert probs[2] == pytest.approx(0.3 * 0.85 ** 2)          # Drama: no overlap
    assert probs[3] == pytest.approx(0.2 * 0.85 ** 2 * 0.8)    # Comedy overlaps slot 1
    assert probs[4] == pytest.approx(0.1 * 0.85 ** 2 * 0.8 ** 2)
    # the [n, P, L] form agrees with the [n, L] form
    orders = np.array([[[0, 1, 2, 3, 4], [2, 0, 3, 1, 4]]])
    both = dqn.order_expected_reward(p, share, orders, d, f)
    assert both[0, 0] == pytest.approx(expected)
    assert both[0, 1] == pytest.approx(dqn.order_expected_reward(p, share, orders[:, 1], d, f)[0])


def test_states_shape_and_content():
    _, g, _ = hand_user()
    used = np.array([[True, True] + [False] * 6])
    s = dqn.build_states(2, used, g)
    assert s.shape == (1, dqn.STATE_DIM) and s.dtype == np.float32
    assert s[0, 0] == pytest.approx(2 / 5)
    assert s[0, 1:9].tolist() == used[0].astype(float).tolist()
    shown = s[0, 9:27]
    assert shown[A] == pytest.approx(2 / 5) and shown[C] == pytest.approx(1 / 5) and shown[D] == 0


def test_action_masking_never_repeats_a_slot():
    rng = np.random.default_rng(0)
    genres = (rng.random((30, dqn.N_CAND, dqn.N_GENRES)) < 0.2).astype(np.float32)
    share, valid = dqn.genre_overlap(genres), np.ones((30, dqn.N_CAND), bool)
    # a Q-function that always prefers the same slots (and random noise) must still give distinct slots
    for q_fn in (lambda s: np.tile(np.arange(dqn.N_CAND, 0, -1, dtype=np.float32), (len(s), 1)),
                 lambda s: rng.normal(size=(len(s), dqn.N_CAND)).astype(np.float32)):
        orders = dqn.policy_orders(q_fn, genres, valid)
        assert orders.shape == (30, dqn.SESSION_LEN)
        assert all(len(set(row)) == dqn.SESSION_LEN for row in orders.tolist())
    for orders in (dqn.diversity_orders(share, valid), dqn.random_orders(30, 1, valid), dqn.greedy_cf_orders(30)):
        assert all(len(set(row)) == dqn.SESSION_LEN for row in orders.tolist())
        assert orders.min() >= 0 and orders.max() < dqn.N_CAND
    assert len(dqn.all_orders()) == 6720


def test_analytic_reward_matches_monte_carlo():
    p, g, share = hand_user()
    d, f = 0.85, 0.8
    order = [3, 0, 2, 1, 4]
    rng = np.random.default_rng(0)
    n_sim = 200_000
    clicks = np.zeros(n_sim)
    used = np.zeros((1, dqn.N_CAND), bool)
    for t, a in enumerate(order):
        prob = dqn.step_click_probs(p, share, used, t, d, f)[0, a]
        clicks += rng.random(n_sim) < prob
        used[0, a] = True
    analytic = dqn.order_expected_reward(p, share, np.array([order]), d, f)[0]
    assert clicks.mean() == pytest.approx(analytic, abs=4 * clicks.std() / np.sqrt(n_sim))


def test_baselines_on_synthetic_env():
    # equal click probs; slots 0-2 all Action, slots 3-7 each a distinct genre
    p = np.full((1, dqn.N_CAND), 0.5)
    g = _genres([[A], [A], [A], [C], [D], [H], [1], [2]])
    share, valid = dqn.genre_overlap(g), np.ones((1, dqn.N_CAND), bool)
    assert dqn.greedy_cf_orders(1).tolist() == [[0, 1, 2, 3, 4]]
    assert dqn.diversity_orders(share, valid).tolist() == [[0, 3, 4, 5, 6]]
    cfg = _cfg(dqn_position_decay=0.85, dqn_fatigue=0.5)
    base = dqn.baseline_rewards({"p": p, "share": share, "valid": valid}, cfg)
    assert base["diversity_heuristic"] > base["random"] > base["greedy_cf"]
    assert base["oracle"] >= max(base["diversity_heuristic"], base["greedy_cf"], base["random"]) - 1e-12
    # without fatigue the CF order is optimal when p is decreasing
    p2 = np.linspace(0.8, 0.1, dqn.N_CAND)[None]
    base2 = dqn.baseline_rewards({"p": p2, "share": share, "valid": valid}, _cfg(dqn_fatigue=1.0))
    assert base2["greedy_cf"] == pytest.approx(base2["oracle"])


def test_build_session_data_signal_and_padding():
    env = synthetic_env(n_users=6)
    env["probs"][0, 0, :] = 0.01             # user 0: no signal in the cf arm
    env["lists"][1, 0, 7] = -1               # padded slot
    gmh = synthetic_genres(40)
    data = dqn.build_session_data(env, gmh, 0.01)
    assert not data["signal"][0] and data["signal"][1:].all()
    assert data["p"].shape == (6, 8) and data["genres"].shape == (6, 8, 18)
    assert data["p"][1, 7] == 0 and data["genres"][1, 7].sum() == 0
    assert not data["valid"][1, 7] and data["valid"].sum() == 6 * 8 - 1


def test_padded_slots_are_shown_last():
    # 3 real candidates that all share Action + 5 padded slots (p = 0, no genres, valid False):
    # without the mask the zero-overlap padded slots would beat the overlapping real ones
    p = np.array([[0.5, 0.4, 0.3, 0, 0, 0, 0, 0]])
    g = _genres([[A], [A], [A], [], [], [], [], []])
    share, valid = dqn.genre_overlap(g), np.array([[True] * 3 + [False] * 5])
    assert dqn.diversity_orders(share, valid)[0, :3].tolist() == [0, 1, 2]
    for s in range(5):
        assert set(dqn.random_orders(1, s, valid)[0, :3].tolist()) == {0, 1, 2}
    prefer_padding = lambda s: np.tile(np.arange(dqn.N_CAND, dtype=np.float32), (len(s), 1))  # noqa: E731
    orders = dqn.policy_orders(prefer_padding, g, valid)
    assert orders[0, :3].tolist() == [2, 1, 0] and len(set(orders[0].tolist())) == dqn.SESSION_LEN
    data = {"p": p, "genres": g, "share": share, "valid": valid}
    assert dqn.dqn_reward(prefer_padding, data, _cfg()) > 0.5   # all three real items shown


# ----------------------------------------------------------------------------- pipeline round trip
def synthetic_genres(n_items: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    g = np.zeros((n_items, dqn.N_GENRES), np.float32)
    for i in range(n_items):
        g[i, rng.choice(dqn.N_GENRES, size=rng.integers(1, 3), replace=False)] = 1.0
    return g


def synthetic_env(n_users: int = 16, n_items: int = 40, seed: int = 0) -> dict:
    """bandit_env-like dict: random lists, mostly click_unrated with a few liked items per user."""
    rng = np.random.default_rng(seed)
    lists = np.stack([np.stack([rng.permutation(n_items)[:10] for _ in range(4)]) for _ in range(n_users)])
    probs = np.full(lists.shape, 0.01, np.float32)
    for u in range(n_users):
        probs[u, 0, rng.choice(8, size=2, replace=False)] = rng.choice([0.45, 0.75], size=2)
    return {"user_idx": np.arange(100, 100 + n_users), "lists": lists.astype(np.int32), "probs": probs,
            "expected": probs.mean(axis=2), "contexts": np.ones((n_users, 5), np.float32),
            "arm_names": np.array(["cf", "content", "cluster_popular", "trending"], dtype="U20")}


def _write_artifacts(tmp_path: Path) -> dict:
    models, processed, figures = tmp_path / "models", tmp_path / "processed", tmp_path / "figures"
    models.mkdir()
    processed.mkdir()
    np.savez(models / "bandit_env.npz", **synthetic_env())
    np.savez(processed / "item_features.npz", genre_multi_hot=synthetic_genres(40))
    return {"models_dir": models, "processed_dir": processed, "figures_dir": figures}


def test_dqn_learns_genre_aware_order():
    """The trained greedy policy reaches the optimum on a toy env where no fixed order is optimal.

    All p = 0.9 and fatigue 0.1. Half the users have slots 0-2 sharing a genre, the other half slots
    3-5; the rest are distinct genres. No set of 5 slots avoids overlap for both groups, so the
    net must read the candidate genres; the oracle (= diversity heuristic here) shows 5 distinct
    genres. An untrained net, or one whose update is broken (e.g. a flipped loss sign), fails.
    """
    torch = pytest.importorskip("torch")
    torch.set_num_threads(2)
    ga = _genres([[A], [A], [A], [C], [D], [H], [1], [2]])
    gb = _genres([[C], [D], [H], [A], [A], [A], [1], [2]])
    genres = np.concatenate([ga, gb] * 10)
    data = {"p": np.full((20, dqn.N_CAND), 0.9), "genres": genres, "share": dqn.genre_overlap(genres),
            "valid": np.ones((20, dqn.N_CAND), bool)}
    cfg = _cfg(dqn_episodes=300, dqn_warmup=32, dqn_batch=32, dqn_fatigue=0.1, dqn_eval_every=50, seed=2)
    net, hist = dqn.train_dqn(data, cfg, log=lambda m: None)
    base = dqn.baseline_rewards(data, cfg)
    final = dqn.dqn_reward(lambda s: dqn.q_values_np(dqn.export_weights(net), s), data, cfg)
    assert final == pytest.approx(hist["eval_train"][-1], abs=1e-4)   # numpy export = torch policy
    assert base["diversity_heuristic"] == pytest.approx(base["oracle"])
    assert final >= base["oracle"] - 0.05, (final, base)
    assert final >= base["random"] + 0.3 and final >= hist["eval_train"][0] + 0.3, (final, hist["eval_train"])


def test_train_and_eval_round_trip(tmp_path):
    torch = pytest.importorskip("torch")
    torch.set_num_threads(2)
    ctx = _write_artifacts(tmp_path)
    cfg = _cfg(dqn_episodes=80, dqn_warmup=32, dqn_batch=32, dqn_eval_every=40, dqn_target_sync=50)
    metrics = dqn.train_step(cfg, ctx)
    keys = {"dqn", "greedy_cf", "random", "diversity_heuristic", "n_train_users", "n_eval_users",
            "episodes", "session_len"}
    assert keys <= set(metrics)
    assert metrics["n_train_users"] == 8 and metrics["n_eval_users"] == 8
    assert metrics["n_users_total"] == 16 and metrics["n_users_with_signal"] == 16
    assert metrics["episodes"] == 80 and metrics["session_len"] == 5
    for k in ("dqn", "greedy_cf", "random", "diversity_heuristic", "oracle"):
        assert np.isfinite(metrics[k]) and 0 <= metrics[k] <= metrics["oracle"] + 1e-4
    models = ctx["models_dir"]
    assert (models / "dqn.npz").exists()
    results = json.loads((models / "dqn_results.json").read_text(encoding="utf-8"))
    assert results["metrics"] == metrics and results["n_users_with_signal"] == 16
    assert results["eval_curve"]["episode"] == [0, 40, 80]
    out = dqn.eval_step(cfg, ctx)
    assert out == metrics
    assert (ctx["figures_dir"] / "dqn_training.png").exists()
    # the saved numpy weights reproduce the torch forward pass
    w = dict(np.load(models / "dqn.npz"))
    net = dqn.build_qnet(dqn.STATE_DIM, cfg.dqn_hidden)
    layers = [m for m in net if hasattr(m, "weight")]
    with torch.no_grad():
        for k, m in enumerate(layers):
            m.weight.copy_(torch.from_numpy(w[f"W{k}"]))
            m.bias.copy_(torch.from_numpy(w[f"b{k}"]))
        s = np.random.default_rng(0).random((5, dqn.STATE_DIM)).astype(np.float32)
        assert np.allclose(net(torch.from_numpy(s)).numpy(), dqn.q_values_np(w, s), atol=1e-5)
    # train and eval users are disjoint and the reported numbers come from the eval users
    tr_ids, ev_ids = w["train_user_idx"], w["eval_user_idx"]
    assert not set(tr_ids.tolist()) & set(ev_ids.tolist())
    assert len(tr_ids) + len(ev_ids) == metrics["n_users_with_signal"]
    data = dqn.build_session_data(synthetic_env(), synthetic_genres(40), cfg.click_unrated)
    evald = dqn.subset(data, np.flatnonzero(np.isin(data["user_idx"], ev_ids)))
    assert evald["user_idx"].tolist() == ev_ids.tolist()
    q_np = lambda s: dqn.q_values_np(w, s)  # noqa: E731
    assert metrics["dqn"] == pytest.approx(dqn.dqn_reward(q_np, evald, cfg), abs=1e-4)
    base = dqn.baseline_rewards(evald, cfg)
    for k in ("greedy_cf", "random", "diversity_heuristic", "oracle"):
        assert metrics[k] == pytest.approx(base[k], abs=1e-4)


def test_steps_return_none_when_artifacts_missing(tmp_path):
    ctx = {"models_dir": tmp_path / "m", "processed_dir": tmp_path / "p", "figures_dir": tmp_path / "f"}
    assert dqn.eval_step(_cfg(), ctx) is None
    assert dqn.train_step(_cfg(), ctx) is None
    assert not (tmp_path / "m" / "dqn.npz").exists()


def test_import_does_not_load_torch():
    env = {**os.environ, "PYTHONPATH": str(SRC)}
    code = "import sys, cinematch.dqn; assert 'torch' not in sys.modules, 'torch loaded'"
    res = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
