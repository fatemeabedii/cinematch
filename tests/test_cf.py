"""Tests for cinematch.cf on small self-contained synthetic data (planted low-rank ratings)."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cinematch import cf, data  # noqa: E402
from cinematch.config import GENRES, Config  # noqa: E402
from cinematch.metrics import rmse_eval  # noqa: E402

N_USERS, N_ITEMS, PER_USER = 300, 200, 60


@pytest.fixture
def few_threads():
    """Cap torch threads for the training tests and restore the previous setting afterwards."""
    import torch
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def make_split(seed: int = 0) -> pd.DataFrame:
    """ratings_split-like frame (u, i, rating, timestamp, split) from a rank-3 model + biases + noise."""
    rng = np.random.default_rng(seed)
    P = rng.normal(0, 0.7, (N_USERS, 3))
    Q = rng.normal(0, 0.7, (N_ITEMS, 3))
    bu, bi = rng.normal(0, 0.3, N_USERS), rng.normal(0, 0.3, N_ITEMS)
    rows = []
    for u in range(N_USERS):
        items = rng.choice(N_ITEMS, PER_USER, replace=False)
        r = np.clip(3.5 + bu[u] + bi[items] + Q[items] @ P[u] + rng.normal(0, 0.2, PER_USER), 1, 5)
        rows += [(u, int(i), float(x)) for i, x in zip(items, r)]
    df = pd.DataFrame(rows, columns=["u", "i", "rating"])
    df["rating"] = df["rating"].astype(np.float32)
    df["timestamp"] = rng.integers(0, 400 * 86400, len(df))
    df["split"] = rng.choice(["train", "val", "test"], size=len(df), p=[0.8, 0.1, 0.1])
    return df


def tiny_cfg(**kw) -> Config:
    cfg = Config(fast=True)
    cfg.cf_epochs, cfg.cf_batch, cfg.cf_lr, cfg.cf_reg, cfg.cf_dim, cfg.cf_patience = 5, 256, 0.01, 0.01, 8, 3
    cfg.eval_users = 50
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


def _data(df) -> dict:
    return {"ratings": df, "stats": {"n_users": N_USERS, "n_items": N_ITEMS}}


def test_baseline_beats_global_mean():
    df = make_split()
    R = data.train_matrix(df, N_USERS, N_ITEMS, "train")
    base = cf.BiasBaseline().fit(R, reg=5.0, iters=10)
    val = df[df["split"] == "val"]
    mu = float(df.loc[df["split"] == "train", "rating"].mean())
    assert rmse_eval(base.predict, val) < rmse_eval(lambda u, i: np.full(len(u), mu), val) - 0.02
    assert base.score_all(np.array([0, 5])).shape == (2, N_ITEMS)
    np.testing.assert_allclose(base.score_all(np.array([3]))[0, 7], base.predict([3], [7])[0], rtol=1e-6)


def _model_val_rmse(model, val: pd.DataFrame) -> float:
    """Val RMSE (clipped, like train_mf) recomputed from the returned model's weights."""
    P, Q = model.user_emb.weight.detach().numpy(), model.item_emb.weight.detach().numpy()
    bu = model.user_bias.weight.detach().numpy().ravel()
    bi = model.item_bias.weight.detach().numpy().ravel()
    mu = float(model.mu)
    return rmse_eval(lambda u, i: mu + bu[u] + bi[i] + (P[u] * Q[i]).sum(1), val)


@pytest.mark.usefixtures("few_threads")
def test_mf_beats_baseline():
    df = make_split()
    cfg = tiny_cfg()
    model, hist = cf.train_mf(_data(df), cfg)
    assert len(hist["val_rmse"]) <= cfg.cf_epochs and 1 <= hist["best_epoch"] <= len(hist["val_rmse"])
    base = cf.BiasBaseline().fit(data.train_matrix(df, N_USERS, N_ITEMS), reg=5.0, iters=10)
    val = df[df["split"] == "val"]
    mf_rmse = _model_val_rmse(model, val)
    assert mf_rmse < rmse_eval(base.predict, val)
    assert mf_rmse == pytest.approx(min(hist["val_rmse"]), abs=1e-3)


@pytest.mark.usefixtures("few_threads")
def test_mf_early_stops_and_restores_best_weights():
    """No regularisation + high lr overfits, so val RMSE turns up: training must stop after
    `cf_patience` bad epochs and hand back the best epoch's weights, not the last ones."""
    df = make_split()
    cfg = tiny_cfg(cf_reg=0.0, cf_dim=32, cf_lr=0.05, cf_epochs=10, cf_patience=2)
    model, hist = cf.train_mf(_data(df), cfg)
    best = hist["best_epoch"]
    assert 1 <= best < len(hist["val_rmse"]) < cfg.cf_epochs  # stopped early, after the best epoch
    assert len(hist["val_rmse"]) == best + cfg.cf_patience
    val_rmse = _model_val_rmse(model, df[df["split"] == "val"])
    assert val_rmse == pytest.approx(hist["val_rmse"][best - 1], abs=1e-4)
    assert abs(val_rmse - hist["val_rmse"][-1]) > 1e-3  # last-epoch weights were discarded


@pytest.mark.usefixtures("few_threads")
def test_export_load_roundtrip(tmp_path):
    df = make_split()
    model, _ = cf.train_mf(_data(df), tiny_cfg(cf_epochs=1))
    cf.export_mf(model, tmp_path / "cf_mf.npz")
    loaded = cf.load_cf(tmp_path)
    assert set(loaded) == {"P", "Q", "bu", "bi", "mu"}
    assert loaded["P"].shape == (N_USERS, 8) and loaded["Q"].shape == (N_ITEMS, 8)
    import torch
    u, i = torch.tensor([0, 1, 2]), torch.tensor([5, 6, 7])
    with torch.no_grad():
        expected = model(u, i).numpy()
    got = loaded["mu"] + loaded["bu"][[0, 1, 2]] + loaded["bi"][[5, 6, 7]] + \
        (loaded["P"][[0, 1, 2]] * loaded["Q"][[5, 6, 7]]).sum(1)
    np.testing.assert_allclose(got, expected, atol=1e-5)
    np.testing.assert_allclose(cf.score_users(loaded, loaded["P"][[1]], loaded["bu"][[1]])[0, 6], got[1], atol=1e-5)

    base = cf.BiasBaseline().fit(data.train_matrix(df, N_USERS, N_ITEMS), reg=5.0, iters=3)
    cf.export_baseline(base, tmp_path / "cf_baseline.npz")
    lb = cf.load_baseline(tmp_path)
    assert lb["mu"] == pytest.approx(base.mu, abs=1e-6)
    np.testing.assert_allclose(lb["bi"], base.bi)


def test_fold_in_recovers_vector():
    rng = np.random.default_rng(1)
    d = 8
    model = {"Q": rng.normal(0, 1, (N_ITEMS, d)).astype(np.float32),
             "bi": rng.normal(0, 0.3, N_ITEMS).astype(np.float32), "mu": 3.5}
    p_true = rng.normal(0, 0.5, d)
    items = rng.choice(N_ITEMS, 40, replace=False)
    r = model["mu"] + 0.2 + model["bi"][items] + model["Q"][items] @ p_true + rng.normal(0, 0.05, 40)
    p, bu = cf.fold_in_user(dict(zip(items.tolist(), r.tolist())), model, reg=1.0)
    cos = p @ p_true / (np.linalg.norm(p) * np.linalg.norm(p_true))
    assert cos > 0.95 and abs(bu - 0.2) < 0.2
    p0, bu0 = cf.fold_in_user({}, model)
    assert p0.shape == (d,) and not p0.any() and bu0 == 0.0


def test_popularity_scores_hand_computed():
    day = 86400
    df = pd.DataFrame({"i": [0, 0, 1, 2, 2, 3], "rating": [5.0, 4.0, 2.0, 3.0, 3.0, 5.0],
                       "timestamp": [0, 200 * day, 10 * day, 150 * day, 199 * day, 200 * day],
                       "split": ["train", "train", "train", "train", "train", "test"]})
    pop = cf.popularity_scores(df, 5)
    assert set(pop) == {"score", "count", "trending"}
    np.testing.assert_allclose(pop["count"], [2, 1, 2, 0, 0])
    np.testing.assert_allclose(pop["score"], [2 * 2.0, 0, 2 * 0.5, 0, 0])
    np.testing.assert_allclose(pop["trending"], [1, 0, 2, 0, 0])  # train rows with ts >= 200 - 90 days


def _raw_frames(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(0)
    ratings = pd.DataFrame({"user_id": df["u"] + 1, "movie_id": df["i"] + 1, "rating": df["rating"].round().clip(1, 5),
                            "timestamp": df["timestamp"] + 978_300_000})
    ratings["rating"] = ratings["rating"].astype(np.float32)
    movies = pd.DataFrame({"movie_id": np.arange(1, N_ITEMS + 1), "title": [f"Movie {k} (1995)" for k in range(N_ITEMS)],
                           "genres": [rng.choice(GENRES) for _ in range(N_ITEMS)]})
    return ratings, movies


@pytest.mark.usefixtures("few_threads")
def test_train_and_eval_step_on_tiny_ctx(tmp_path):
    cfg = tiny_cfg(cold_item_min_ratings=5, new_user_min_train_ratings=5, new_user_seed_ratings=3,
                   cf_rank_pop_betas=(0.0, 1.0), knn_neighbors=(5, 20), knn_pop_gammas=(0.0, 0.5))
    ratings, movies = _raw_frames(make_split())
    ctx = {"processed_dir": tmp_path / "processed", "models_dir": tmp_path / "models",
           "figures_dir": tmp_path / "figures", "flagged_user_ids": [1, 2]}
    prepared = data.prepare_data(cfg, [1, 2], processed_dir=ctx["processed_dir"], ratings=ratings, movies=movies)
    ctx["models_dir"].mkdir()
    np.save(ctx["models_dir"] / "eval_users.npy", data.eval_users_sample(prepared["ratings"], 50, cfg.seed, [0, 1]))

    summary = cf.train_step(cfg, ctx)  # no ctx["data"]: loads processed data itself
    assert summary["best_epoch"] >= 1
    for f in ("cf_baseline.npz", "cf_mf.npz", "popularity.npz", "cf_history.json"):
        assert (ctx["models_dir"] / f).exists()
    with np.load(ctx["models_dir"] / "popularity.npz") as z:
        assert sorted(z.files) == ["count", "score", "trending"]
        assert z["score"][prepared["item_features"]["is_cold"]].sum() == 0
    hist = json.loads((ctx["models_dir"] / "cf_history.json").read_text(encoding="utf-8"))
    assert {"train_rmse", "val_rmse", "best_epoch"} <= set(hist)

    # flagged users (ids 1, 2) are left out of every fitted model
    flagged = data.flagged_indices(prepared["index_maps"])
    assert sorted(flagged.tolist()) == [0, 1]
    assert not cf.load_baseline(ctx["models_dir"])["bu"][flagged].any()
    assert (np.linalg.norm(cf.load_cf(ctx["models_dir"])["P"][flagged], axis=1) < 0.5).all()  # untouched init
    rs = prepared["ratings"]
    kept = rs[(rs["split"] == "train") & ~rs["u"].isin(flagged)]
    with np.load(ctx["models_dir"] / "popularity.npz") as z:
        np.testing.assert_allclose(z["count"], np.bincount(kept["i"].to_numpy(), minlength=len(z["count"])))

    # ranking rules chosen on validation: cf_ranking.json + the chosen item-kNN matrix
    ranking = json.loads((ctx["models_dir"] / "cf_ranking.json").read_text(encoding="utf-8"))
    assert ranking["beta"] in cfg.cf_rank_pop_betas and ranking["knn_k"] in cfg.knn_neighbors
    assert ranking["knn_gamma"] in cfg.knn_pop_gammas
    assert set(ranking["val_ndcg"]["beta"]) == {str(float(b)) for b in cfg.cf_rank_pop_betas}
    assert len(ranking["val_ndcg"]["knn"]) == len(cfg.knn_neighbors) * len(cfg.knn_pop_gammas)
    S = cf.load_item_knn(ctx["models_dir"])
    assert S.shape == (N_ITEMS, N_ITEMS) and (S.getnnz(axis=1) <= ranking["knn_k"]).all()

    out = cf.eval_step(cfg, ctx)
    assert set(out) == {"popularity", "bias_baseline", "cf_mf", "cf_mf_pop", "item_knn", "new_users"}
    assert out["popularity"]["rmse"] is None
    for name in ("bias_baseline", "cf_mf"):
        assert np.isfinite(out[name]["rmse"]) and 0.3 < out[name]["rmse"] < 1.5
    rows = [m for name, m in out.items() if name != "new_users"] + list(out["new_users"]["strategies"].values())
    for m in rows:
        for key in ("precision_at_10", "recall_at_10", "ndcg_at_10"):
            assert 0.0 <= m[key] <= 1.0
    assert out["cf_mf_pop"]["beta"] == ranking["beta"]
    assert out["item_knn"]["k"] == ranking["knn_k"] and out["item_knn"]["gamma"] == ranking["knn_gamma"]
    nu = out["new_users"]
    assert set(nu) == {"n_users", "n_seed_ratings", "strategies"} and nu["n_seed_ratings"] == 3
    assert set(nu["strategies"]) == {"popularity", "mf_foldin", "mf_foldin_pop", "item_knn"}
    assert 0 < nu["n_users"] <= cfg.new_user_eval_users
    assert out["cf_mf"]["dim"] == cfg.cf_dim and out["cf_mf"]["best_epoch"] == hist["best_epoch"]
    assert (ctx["figures_dir"] / "cf_training.png").exists()


def test_eval_step_returns_none_without_artifacts(tmp_path):
    ctx = {"models_dir": tmp_path / "models", "processed_dir": tmp_path / "p", "figures_dir": tmp_path / "f"}
    assert cf.eval_step(tiny_cfg(), ctx) is None


def test_import_does_not_load_torch():
    code = "import sys, cinematch.cf; assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True, env={**os.environ, "PYTHONPATH": str(SRC)})


# ----------------------------------------------------------------------------- ranking rules
def test_rank_scores_adds_log_popularity():
    rng = np.random.default_rng(0)
    model = {"P": rng.normal(size=(4, 3)).astype(np.float32), "Q": rng.normal(size=(6, 3)).astype(np.float32),
             "bu": rng.normal(size=4).astype(np.float32), "bi": rng.normal(size=6).astype(np.float32), "mu": 3.5}
    count = np.array([0, 1, 10, 100, 1000, 5], dtype=np.float32)
    base = cf.score_users(model, model["P"][[1, 2]], model["bu"][[1, 2]])
    got = cf.rank_scores(model, model["P"][[1, 2]], model["bu"][[1, 2]], count, 0.7)
    assert got.shape == (2, 6)
    np.testing.assert_allclose(got, base + 0.7 * np.log1p(count)[None, :], rtol=1e-6)
    np.testing.assert_allclose(cf.rank_scores(model, model["P"][1], model["bu"][1], count, 0.0), base[[0]], rtol=1e-6)


def test_item_knn_cosine_topk_and_roundtrip(tmp_path):
    rng = np.random.default_rng(3)
    X = (rng.random((40, 12)) < 0.3).astype(np.float32)
    X[:, 11] = 0  # an item nobody liked has no neighbours
    liked = sp.csr_matrix(X)
    full = cf.build_item_knn(liked, k=100)  # k >= n_items: keep everything
    norms = np.sqrt(X.sum(0))
    expected = (X.T @ X) / np.maximum(np.outer(norms, norms), 1e-12)
    np.fill_diagonal(expected, 0)
    assert full.dtype == np.float32 and full.shape == (12, 12)
    np.testing.assert_allclose(full.toarray(), expected, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(full.toarray(), full.toarray().T, atol=1e-6)  # cosine is symmetric
    assert not full.diagonal().any() and full[11].nnz == 0 and full[:, 11].nnz == 0

    S = cf.build_item_knn(liked, k=3)
    assert (S.getnnz(axis=1) <= 3).all() and not S.diagonal().any()
    for i in range(11):  # each row keeps its 3 largest similarities
        kept = np.sort(S[i].data)[::-1]
        np.testing.assert_allclose(kept, np.sort(expected[i])[::-1][:len(kept)], rtol=1e-5)

    cf.save_item_knn(S, tmp_path / "item_knn.npz")
    with np.load(tmp_path / "item_knn.npz") as z:
        assert sorted(z.files) == ["data", "indices", "indptr", "shape"]
    loaded = cf.load_item_knn(tmp_path)
    np.testing.assert_allclose(loaded.toarray(), S.toarray())

    scores = cf.knn_scores(S, liked[[0, 5]])
    assert isinstance(scores, np.ndarray) and scores.shape == (2, 12)
    np.testing.assert_allclose(scores, X[[0, 5]] @ S.toarray(), rtol=1e-5)


def test_knn_top_contributor_and_new_user_fallbacks():
    S = sp.csr_matrix(np.array([[0, 0.2, 0.9, 0], [0.2, 0, 0.5, 0], [0.9, 0.5, 0, 0], [0, 0, 0, 0]], np.float32))
    assert cf.knn_top_contributor(S, np.array([0, 1]), 2) == 0
    assert cf.knn_top_contributor(S, np.array([1]), 2) == 1
    assert cf.knn_top_contributor(S, np.array([0, 1]), 3) == -1
    assert cf.knn_top_contributor(S, np.array([], dtype=np.int64), 2) == -1

    count, pop = np.array([1, 2, 3, 4.0]), np.array([9, 8, 7, 6.0])
    np.testing.assert_array_equal(cf.seed_liked_items({0: 5, 1: 2}), [0])
    np.testing.assert_array_equal(cf.seed_liked_items({0: 3, 1: 2}), [0])  # nothing >= 4: fall back to >= 3
    assert len(cf.seed_liked_items({0: 1, 1: 2})) == 0
    got = cf.new_user_knn_scores(S, {0: 5.0, 3: 1.0}, count, 0.5, pop)
    np.testing.assert_allclose(got, S.toarray()[0] + 0.5 * np.log1p(count), rtol=1e-6)
    np.testing.assert_allclose(cf.new_user_knn_scores(S, {0: 1.0}, count, 0.5, pop), pop)  # nothing liked


def _block_data(n_blocks: int, mf_knows_block: bool, seed: int = 0):
    """Users in `n_blocks` taste blocks of 10 items (+10 filler items nobody likes): each user likes 5
    block items in TRAIN and 3 others in VAL. MF either knows the block (one-hot P, Q) or only has an
    item bias that prefers the filler items."""
    rng = np.random.default_rng(seed)
    n_items, filler = 10 * n_blocks + 10, np.arange(10 * n_blocks, 10 * n_blocks + 10)
    rows, user_block = [], []
    for u in range(20 * n_blocks):
        b = u % n_blocks
        items = 10 * b + rng.permutation(10)
        rows += [(u, int(i), 5.0, "train") for i in items[:5]] + [(u, int(i), 5.0, "val") for i in items[5:8]]
        rows.append((u, int(rng.choice(filler)), 2.0, "train"))
        user_block.append(b)
    rs = pd.DataFrame(rows, columns=["u", "i", "rating", "split"])
    n_users = len(user_block)
    Q = np.zeros((n_items, n_blocks), np.float32)
    Q[np.arange(10 * n_blocks), np.arange(10 * n_blocks) // 10] = 1
    P = np.eye(n_blocks, dtype=np.float32)[user_block] * (2.0 if mf_knows_block else 0.0)
    bi = np.zeros(n_items, np.float32)
    if not mf_knows_block:
        bi[filler] = 0.5
    mf = {"P": P, "Q": Q, "bu": np.zeros(n_users, np.float32), "bi": bi, "mu": 3.5}
    data_dict = {"ratings": rs, "eval_users": np.arange(n_users), "warm_mask": np.ones(n_items, bool),
                 "train_items": data.user_train_items(rs, n_users),
                 "stats": {"n_users": n_users, "n_items": n_items}}
    return data_dict, mf, cf.liked_matrix(rs, n_users, n_items)


def test_select_ranking_picks_planted_best_beta_on_popular_truth():
    """MF only knows a bias toward unliked filler items; the relevant items are the popular ones."""
    d, mf, liked = _block_data(n_blocks=1, mf_knows_block=False)
    count = np.where(np.arange(liked.shape[1]) < 10, 100.0, 1.0)
    cfg = tiny_cfg(cf_rank_pop_betas=(0.0, 50.0, 0.001), knn_neighbors=(3,), knn_pop_gammas=(0.0,))
    ranking, S = cf.select_ranking(cfg, d, mf, liked, count)
    assert ranking["beta"] == 50.0
    assert ranking["val_ndcg"]["beta"]["50.0"] > ranking["val_ndcg"]["beta"]["0.0"]
    assert (S.getnnz(axis=1) <= 3).all()


def test_select_ranking_picks_planted_best_personal_over_popular():
    """Three taste blocks; popularity favours block 0 only, so the personal scores (beta = gamma = 0) win."""
    d, mf, liked = _block_data(n_blocks=3, mf_knows_block=True)
    count = np.where(np.arange(liked.shape[1]) < 10, 1000.0, 1.0)
    cfg = tiny_cfg(cf_rank_pop_betas=(50.0, 0.0), knn_neighbors=(9, 4), knn_pop_gammas=(50.0, 0.0))
    ranking, S = cf.select_ranking(cfg, d, mf, liked, count)
    assert ranking["beta"] == 0.0 and ranking["knn_gamma"] == 0.0
    assert ranking["val_ndcg"]["knn"][f"{ranking['knn_k']},0.0"] > ranking["val_ndcg"]["knn"]["9,50.0"]
    assert (S.getnnz(axis=1) <= ranking["knn_k"]).all()


def test_select_ranking_ignores_test_split():
    """TEST rows that reward popularity (every user 'likes' all of popular block 0) must not change the
    choice: selection reads only train and val, so the personal scores still win and the dict is identical."""
    d, mf, liked = _block_data(n_blocks=3, mf_knows_block=True)
    count = np.where(np.arange(liked.shape[1]) < 10, 1000.0, 1.0)
    cfg = tiny_cfg(cf_rank_pop_betas=(50.0, 0.0), knn_neighbors=(9, 4), knn_pop_gammas=(50.0, 0.0))
    before, _ = cf.select_ranking(cfg, d, mf, liked, count)
    test_rows = pd.DataFrame([(u, i, 5.0, "test") for u in d["eval_users"] for i in range(10)],
                             columns=["u", "i", "rating", "split"])
    d["ratings"] = pd.concat([d["ratings"], test_rows], ignore_index=True)
    after, _ = cf.select_ranking(cfg, d, mf, liked, count)
    assert after == before
    assert after["beta"] == 0.0 and after["knn_gamma"] == 0.0
