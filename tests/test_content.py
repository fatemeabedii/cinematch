"""Tests for the two-tower content model (self-contained synthetic data)."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cinematch import content
from cinematch.config import GENRES, Config

torch = pytest.importorskip("torch")
SRC = Path(__file__).resolve().parents[1] / "src"


def _item_feats(n_items=12, seed=0):
    rng = np.random.default_rng(seed)
    g = (rng.random((n_items, 18)) < 0.2).astype(np.float32)
    g[np.arange(n_items), rng.integers(0, 18, n_items)] = 1.0
    return {"genre_multi_hot": g, "year_scaled": rng.random(n_items).astype(np.float32),
            "title_svd": rng.normal(size=(n_items, 16)).astype(np.float32),
            "is_cold": np.zeros(n_items, dtype=bool)}


def _tiny_raw(n_users=120, n_movies=80, seed=1):
    """Planted-preference ratings: users like some genres, so content features carry signal."""
    rng = np.random.default_rng(seed)
    movies = pd.DataFrame({
        "movie_id": np.arange(1, n_movies + 1),
        "title": [f"Movie {k} ({1950 + k % 50})" for k in range(n_movies)],
        "genres": ["|".join(rng.choice(GENRES, size=rng.integers(1, 4), replace=False)) for _ in range(n_movies)],
    })
    g = np.array([[gen in s.split("|") for gen in GENRES] for s in movies["genres"]], dtype=float)
    taste = rng.normal(size=(n_users, 18))
    rows = []
    for u in range(n_users):
        items = rng.choice(n_movies, size=rng.integers(25, 50), replace=False)
        for t, i in enumerate(items):
            r = 3.4 + 0.8 * taste[u] @ g[i] / max(g[i].sum(), 1) + rng.normal(0, 0.5)
            rows.append((u + 1, i + 1, float(np.clip(np.rint(r), 1, 5)), 1_000_000 + 3600 * t))
    ratings = pd.DataFrame(rows, columns=["user_id", "movie_id", "rating", "timestamp"])
    ratings["rating"] = ratings["rating"].astype(np.float32)
    return ratings, movies


# ----------------------------------------------------------------------------- features
def test_item_matrix_shape():
    X = content.item_matrix(_item_feats())
    assert X.shape == (12, content.ITEM_FEAT_DIM) and X.dtype == np.float32


def test_user_profile_known_values_and_empty_user():
    f = _item_feats()
    rs = pd.DataFrame({"u": [0, 0, 1], "i": [2, 5, 3], "rating": [5.0, 1.0, 4.0],
                       "split": ["train", "train", "val"]})
    X = content.user_profile_features(rs, f, n_users=3)
    assert X.shape == (3, content.USER_FEAT_DIM)
    g = f["genre_multi_hot"]
    np.testing.assert_allclose(X[0, :18], (2 * g[2] - 2 * g[5]) / 2, atol=1e-6)
    np.testing.assert_allclose(X[0, 18:36], (g[2] + g[5]) / 2, atol=1e-6)
    np.testing.assert_allclose(X[0, 36:40], [3.0, 2.0, np.log1p(2), (f["year_scaled"][2] + f["year_scaled"][5]) / 2],
                               atol=1e-6)
    assert not X[1].any() and not X[2].any()   # user 1 only has a val row; user 2 has none


def test_profile_from_ratings_matches_matrix_row():
    f = _item_feats()
    rs = pd.DataFrame({"u": [1, 1, 1, 0], "i": [0, 4, 7, 3], "rating": [4.0, 2.0, 5.0, 3.0],
                       "split": ["train"] * 4})
    X = content.user_profile_features(rs, f, n_users=2)
    np.testing.assert_allclose(content.profile_from_ratings({0: 4.0, 4: 2.0, 7: 5.0}, f), X[1], atol=1e-6)


def test_leave_one_out_profile_matches_profile_without_that_row():
    f = _item_feats()
    u = np.array([0, 0, 0, 1, 1, 2])
    i = np.array([1, 4, 9, 2, 3, 5])
    r = np.array([5.0, 2.0, 4.0, 3.0, 1.0, 4.0])
    sums, count = content._user_sums(u, i, r, 3, f)
    loo = content._finalize(sums[u] - content._row_contrib(i, r, f), count[u] - 1.0)
    for row in range(len(u)):
        keep = np.arange(len(u)) != row
        expected = content._profiles(u[keep], i[keep], r[keep], 3, f)[u[row]]
        np.testing.assert_allclose(loo[row], expected, atol=1e-5)
    assert not loo[5].any()   # user 2's only row -> empty profile


# ----------------------------------------------------------------------------- numpy == torch
def test_numpy_forward_matches_torch(tmp_path):
    torch.manual_seed(0)
    model = content.build_two_tower(hidden=16, dim=8, mu=3.6)
    for p in model.parameters():           # non-trivial weights
        torch.nn.init.normal_(p, std=0.3)
    model.eval()
    rng = np.random.default_rng(0)
    item_X = content.item_matrix(_item_feats(20))
    user_raw = rng.normal(2.0, 3.0, size=(7, content.USER_FEAT_DIM)).astype(np.float32)
    mean, std = user_raw.mean(0), user_raw.std(0) + 0.5
    content.export_two_tower(model, mean, std, item_X, user_raw, tmp_path / "two_tower.npz")
    tt = content.load_two_tower(tmp_path)
    for key in ("W_u1", "b_u1", "W_u2", "b_u2", "W_i1", "b_i1", "W_i2", "b_i2", "mu", "feat_mean",
                "feat_std", "item_vectors", "user_feat_raw"):
        assert key in tt
    assert tt["item_vectors"].shape == (20, 8)

    with torch.no_grad():
        ux = torch.from_numpy((user_raw - mean) / std)
        ix = torch.from_numpy(item_X)
        tu, ti = model.user_tower(ux).numpy(), model.item_tower(ix).numpy()
        pair = model(ux[[0, 3, 6]], ix[[1, 2, 19]]).numpy()
    np.testing.assert_allclose(content.item_vectors_np(tt, item_X), ti, atol=1e-5)
    np.testing.assert_allclose(content.user_vector_np(tt, user_raw), tu, atol=1e-5)
    np.testing.assert_allclose(content.user_vector_np(tt, user_raw[2]), tu[2], atol=1e-5)
    scores = content.score_users_tt(tt, content.user_vector_np(tt, user_raw))
    np.testing.assert_allclose(scores[[0, 3, 6], [1, 2, 19]], pair, atol=1e-5)


def test_module_import_does_not_load_torch():
    code = "import sys, cinematch.content; assert 'torch' not in sys.modules"
    env = dict(os.environ, PYTHONPATH=str(SRC))
    subprocess.run([sys.executable, "-c", code], check=True, env=env)


# ----------------------------------------------------------------------------- end-to-end steps
@pytest.fixture
def tiny_dirs(tmp_path):
    from cinematch import data as data_mod

    torch.set_num_threads(2)
    # Config.__post_init__ overwrites the FAST fields (tt_epochs, eval_users), so set them afterwards.
    cfg = Config(fast=True)
    for k, v in dict(cold_item_min_ratings=5, cold_item_frac=0.15, eval_users=60, tt_epochs=10,
                     tt_patience=2, tt_batch=256, tt_hidden=16, tt_dim=8, tt_lr=2e-2).items():
        setattr(cfg, k, v)
    ratings, movies = _tiny_raw()
    dirs = {"processed_dir": tmp_path / "processed", "models_dir": tmp_path / "models",
            "figures_dir": tmp_path / "figures"}
    d = data_mod.prepare_data(cfg, [3, 7], processed_dir=dirs["processed_dir"], ratings=ratings, movies=movies)
    eval_users = data_mod.eval_users_sample(d["ratings"], cfg.eval_users, cfg.seed,
                                            exclude_users=data_mod.flagged_indices(d["index_maps"]))
    dirs["models_dir"].mkdir()
    np.save(dirs["models_dir"] / "eval_users.npy", eval_users)
    return cfg, dirs


def test_eval_step_without_model_returns_none(tmp_path):
    assert content.eval_step(Config(fast=True), {"models_dir": tmp_path}) is None


def test_train_and_eval_steps(tiny_dirs):
    cfg, dirs = tiny_dirs
    out = content.train_step(cfg, dict(dirs))
    models = dirs["models_dir"]
    assert (models / "two_tower.npz").exists()
    hist = json.loads((models / "tt_history.json").read_text())
    assert set(hist) == {"train_rmse", "val_rmse", "best_epoch"}
    assert 1 <= hist["best_epoch"] <= len(hist["val_rmse"]) == len(hist["train_rmse"])
    assert out["best_epoch"] == hist["best_epoch"]
    # Seeded tiny run early-stops after its best epoch, so the restore-best-weights path is exercised.
    assert len(hist["val_rmse"]) < cfg.tt_epochs and hist["best_epoch"] < len(hist["val_rmse"])
    assert hist["train_rmse"][-1] < hist["train_rmse"][0]   # it learns something

    tt = content.load_two_tower(models)
    assert tt["user_feat_raw"].shape[1] == content.USER_FEAT_DIM
    assert tt["item_vectors"].shape == (80, cfg.tt_dim)

    # Best weights are restored: the exported numpy model reproduces the best epoch's val RMSE.
    from cinematch import data as data_mod
    from cinematch.metrics import rmse_eval
    d = data_mod.load_processed(dirs["processed_dir"])
    rs = d["ratings"]
    n_users = len(d["index_maps"]["user_ids"])
    np.testing.assert_allclose(tt["user_feat_raw"],
                               content.user_profile_features(rs, d["item_features"], n_users), atol=1e-6)
    uv = content.user_vector_np(tt, tt["user_feat_raw"])
    val = rs[rs["split"] == "val"]
    v = rmse_eval(lambda u, i: tt["mu"] + np.einsum("nd,nd->n", uv[u], tt["item_vectors"][i]), val)
    assert v == pytest.approx(hist["val_rmse"][hist["best_epoch"] - 1], abs=1e-4)

    full = content.eval_step(cfg, dict(dirs))
    assert set(full) == {"two_tower", "cold_baselines"}
    assert full["cold_baselines"]["random"]["n_eval_users"] == full["two_tower"]["cold"]["n_eval_users"]
    res = full["two_tower"]
    for key in ("rmse", "precision_at_10", "recall_at_10", "ndcg_at_10"):
        assert np.isfinite(res[key]) and np.isfinite(res["cold"][key])
    assert 0.0 < res["rmse"] < 2.0 and 0.0 < res["cold"]["rmse"] < 2.0
    assert res["cold"]["n_eval_users"] > 0
    assert res["best_epoch"] == hist["best_epoch"]
    assert (dirs["figures_dir"] / "two_tower_training.png").exists()


# ----------------------------------------------------------------------------- cold-start baselines (17.5)
def _loaded(cfg, dirs):
    from cinematch import data as data_mod

    ctx = dict(dirs)
    data_mod.eval_step(cfg, ctx)
    return ctx["data"]


def test_cold_baselines_keys_seeded_random_and_missing_baseline(tiny_dirs):
    cfg, dirs = tiny_dirs
    data = _loaded(cfg, dirs)
    out = content.cold_baselines(cfg, data, dirs["models_dir"])     # no cf_baseline.npz in models_dir
    assert set(out) == {"user_bias", "random", "genre_popularity"}
    assert out["user_bias"] == {"rmse": None}
    assert set(out["random"]) == {"precision_at_10", "recall_at_10", "ndcg_at_10", "n_eval_users", "n_seeds"}
    assert out["random"]["n_seeds"] == content.N_RANDOM_SEEDS
    assert out["random"]["n_eval_users"] > 0
    assert set(out["genre_popularity"]) == {"precision_at_10", "recall_at_10", "ndcg_at_10", "n_eval_users"}
    assert out["genre_popularity"]["n_eval_users"] == out["random"]["n_eval_users"]
    for k in ("precision_at_10", "recall_at_10", "ndcg_at_10"):
        assert np.isfinite(out["random"][k]) and 0.0 <= out["random"][k] <= 1.0
    assert content.cold_baselines(cfg, data, dirs["models_dir"]) == out    # seeded -> identical

    # 17.5 contract rebuilt independently: mean over seeds cfg.seed..cfg.seed+9, cold candidates only.
    from cinematch.config import TOP_K
    from cinematch.metrics import rank_eval
    n_items = len(data["warm_mask"])

    def runs(seeds, cand):
        return [rank_eval(lambda us, rng=np.random.default_rng(s): rng.random((len(us), n_items)),
                          data["eval_users"], data["holdout_cold"], data["train_items"], cand, TOP_K,
                          cfg.eval_rank_batch) for s in seeds]

    def mean4(rs):
        return {k: round(float(np.mean([r[k] for r in rs])), 4)
                for k in ("precision_at_10", "recall_at_10", "ndcg_at_10")}

    cold = ~np.asarray(data["warm_mask"], dtype=bool)
    expected = mean4(runs(range(cfg.seed, cfg.seed + 10), cold))
    got = {k: out["random"][k] for k in expected}
    assert got == expected
    # Discriminative: a single seed, or ranking warm items instead of cold ones, gives other numbers.
    assert got != mean4(runs([cfg.seed], cold))
    assert got != mean4(runs(range(cfg.seed, cfg.seed + 10), ~cold))


def test_cold_baselines_user_bias_rmse(tiny_dirs):
    cfg, dirs = tiny_dirs
    data = _loaded(cfg, dirs)
    n_users, n_items = len(data["index_maps"]["user_ids"]), len(data["warm_mask"])
    bu = np.linspace(-3.0, 3.0, n_users)            # large biases so clipping to [1, 5] matters
    np.savez(dirs["models_dir"] / "cf_baseline.npz", mu=np.array(3.5), bu=bu, bi=np.full(n_items, 9.0))
    cold = data["ratings"][data["ratings"]["split"] == "cold"]
    pred = np.clip(3.5 + bu[cold["u"].to_numpy()], 1, 5)                  # item bias ignored for cold items
    expected = np.sqrt(np.mean((pred - cold["rating"].to_numpy()) ** 2))
    rmse = content.cold_baselines(cfg, data, dirs["models_dir"])["user_bias"]["rmse"]
    assert rmse == pytest.approx(expected, abs=1e-4)


def _planted_cold_world(n_eval=30, seed=0):
    """Warm items 0-9 (0-4 genre 0 = popular, 5-9 genre 1), cold items 10-19 genre 1, 20-29 genre 0,
    30 no genre. Eval users (0..n_eval-1) like 3 random cold genre-0 items; tie-breaking by lower
    index favours the unpopular genre, so only a real genre prior can win. Non-eval users have
    cold rows (split "cold") on genre-1 items that a cold-rating leak would reward."""
    rng = np.random.default_rng(seed)
    n_items, n_users = 31, n_eval + 20
    G = np.zeros((n_items, 18), dtype=np.float32)
    G[[0, 1, 2, 3, 4, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29], 0] = 1.0
    G[[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19], 1] = 1.0
    warm = np.zeros(n_items, dtype=bool)
    warm[:10] = True
    rows = []
    for u in range(n_users):
        for i in range(5):                                 # every user rates the popular warm genre
            rows.append((u, i, 4.0, "train"))
        rows.append((u, 5 + u % 5, 3.0, "train"))          # 1 rating per user on the other genre
    for u in range(n_eval):
        for i in rng.choice(np.arange(20, 30), size=3, replace=False):
            rows.append((u, int(i), 5.0, "cold"))
        rows.append((u, int(rng.integers(10, 20)), 2.0, "cold"))
    for u in range(n_eval, n_users):
        for i in range(10, 20):
            rows.append((u, i, 5.0, "cold"))
    rs = pd.DataFrame(rows, columns=["u", "i", "rating", "split"])
    return _world_dict(rs, G, warm, n_users, np.arange(n_eval))


def _world_dict(rs, G, warm, n_users, eval_users):
    cold = rs[rs["split"] == "cold"]
    train = rs[rs["split"] == "train"]
    return {"ratings": rs, "item_features": {"genre_multi_hot": G}, "warm_mask": warm,
            "eval_users": eval_users,
            "holdout_cold": {int(u): dict(zip(g["i"].tolist(), g["rating"].tolist()))
                             for u, g in cold.groupby("u")},
            "train_items": [np.sort(train.loc[train["u"] == u, "i"].to_numpy()) for u in range(n_users)]}


def test_genre_popularity_scores_known_values():
    d = _planted_cold_world()
    s = content.genre_popularity_scores(d)
    n_users = len(d["train_items"])
    # genre 0: 5 warm movies x n_users ratings each; genre 1: n_users ratings over 5 warm movies.
    pop0, pop1 = float(n_users), n_users / 5
    np.testing.assert_allclose(s[[0, 20, 29]], pop0)
    np.testing.assert_allclose(s[[5, 10, 19]], pop1)
    assert s[30] == 0.0                                    # no genre -> 0
    G = d["item_features"]["genre_multi_hot"].copy()
    G[30, [0, 1]] = 1.0                                    # two genres -> mean of the two
    s2 = content.genre_popularity_scores({**d, "item_features": {"genre_multi_hot": G}})
    assert s2[30] == pytest.approx((pop0 + pop1) / 2)


def test_genre_popularity_beats_random_in_planted_world(tmp_path):
    cfg = Config(fast=True)
    out = content.cold_baselines(cfg, _planted_cold_world(), tmp_path)
    gp, rnd = out["genre_popularity"], out["random"]
    assert set(gp) == {"precision_at_10", "recall_at_10", "ndcg_at_10", "n_eval_users"}
    assert gp["n_eval_users"] == rnd["n_eval_users"] == 30
    assert gp["recall_at_10"] == 1.0 and gp["precision_at_10"] == pytest.approx(0.3, abs=1e-4)
    for k in ("precision_at_10", "recall_at_10", "ndcg_at_10"):
        assert gp[k] > rnd[k]


def test_genre_popularity_ignores_cold_ratings(tmp_path):
    cfg = Config(fast=True)
    d = _planted_cold_world()
    rs = d["ratings"].copy()
    non_eval_cold = (rs["split"] == "cold") & (rs["u"] >= 30)
    rs.loc[non_eval_cold, "rating"] = 1.0
    extra = pd.DataFrame({"u": np.repeat(np.arange(30, 50), 10), "i": np.tile(np.arange(20, 30), 20),
                          "rating": 5.0, "split": "cold"})
    changed = _world_dict(pd.concat([rs, extra], ignore_index=True), d["item_features"]["genre_multi_hot"],
                          d["warm_mask"], len(d["train_items"]), d["eval_users"])
    np.testing.assert_array_equal(content.genre_popularity_scores(changed), content.genre_popularity_scores(d))
    assert (content.cold_baselines(cfg, changed, tmp_path)["genre_popularity"]
            == content.cold_baselines(cfg, d, tmp_path)["genre_popularity"])
