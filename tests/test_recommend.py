"""Tests for the numpy-only Recommender, on artifacts from a tiny FAST training run in tmp dirs."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import make_tiny_movies, make_tiny_ratings, write_raw_dat
from cinematch.config import Config
from cinematch.recommend import (FROM_RATINGS_STRATEGIES, MIN_TRAIN_RATINGS, USER_STRATEGIES,
                                 ModelsMissingError, Recommender)

ROOT = Path(__file__).resolve().parents[1]
N = 10


@pytest.fixture(scope="module")
def tiny_models(tmp_path_factory):
    """Run data/cf/content/clustering train_steps (FAST) on the tiny synthetic data into tmp dirs."""
    import torch

    from cinematch import cf, clustering, content, data
    from cinematch.config import set_seed

    base = tmp_path_factory.mktemp("tiny_models")
    raw_dir = write_raw_dat(make_tiny_ratings(), make_tiny_movies(), base / "ml-1m")
    cfg = Config(fast=True)
    cfg.cold_item_min_ratings, cfg.eval_users = 5, 50
    ctx = {"flagged_user_ids": [], "timings": {}, "raw_dir": raw_dir, "processed_dir": base / "processed",
           "models_dir": base / "models", "figures_dir": base / "figures"}
    old_threads = torch.get_num_threads()
    try:
        for step in (data, cf, content, clustering):
            set_seed(cfg.seed)
            torch.set_num_threads(2)   # set_seed raises it to 8; other agents share the CPU
            step.train_step(cfg, ctx)
    finally:
        torch.set_num_threads(old_threads)
    return ctx


@pytest.fixture(scope="module")
def rec(tiny_models):
    return Recommender(tiny_models["models_dir"], tiny_models["processed_dir"])


@pytest.fixture(scope="module")
def new_user_ratings(rec):
    """Five well-known movies (movie_id keys), mixed ratings."""
    ids = rec.popular_for_rating(5)["movie_id"].tolist()
    return dict(zip(ids, [5, 4, 5, 2, 1]))


def test_models_missing_error_on_empty_dirs(tmp_path):
    with pytest.raises(ModelsMissingError) as exc:
        Recommender(tmp_path / "models", tmp_path / "processed")
    assert isinstance(exc.value, FileNotFoundError)
    missing = " ".join(exc.value.missing)
    for f in ("cf_mf.npz", "two_tower.npz", "clusters.json", "popularity.npz", "movies.parquet",
              "user_train_items.npz"):
        assert f in missing
    assert "train_all.py" in str(exc.value)


def test_models_missing_lists_only_missing_files(tiny_models, tmp_path):
    with pytest.raises(ModelsMissingError) as exc:
        Recommender(tiny_models["models_dir"], tmp_path)
    assert all(Path(p).parent == tmp_path for p in exc.value.missing)
    assert len(exc.value.missing) == 4


@pytest.mark.parametrize("strategy", FROM_RATINGS_STRATEGIES)
def test_recommend_from_ratings_returns_n_distinct_unrated(rec, new_user_ratings, strategy):
    df = rec.recommend_from_ratings(new_user_ratings, strategy=strategy, n=N)
    assert list(df.columns) == ["movie_id", "title", "year", "genres", "score", "why"]
    assert len(df) == N and df["movie_id"].is_unique
    assert not set(df["movie_id"]) & set(new_user_ratings)
    assert np.all(np.isfinite(df["score"])) and np.all(np.diff(df["score"].to_numpy()) <= 1e-12)
    assert all(isinstance(w, str) and w for w in df["why"])
    movies = rec.movies_table().set_index("movie_id")
    assert (movies.loc[df["movie_id"], "title"].to_numpy() == df["title"].to_numpy()).all()
    if strategy != "hybrid":    # cf/content scores are predicted ratings, shown on the 1..5 scale
        assert df["score"].between(1.0, 5.0).all()
    if strategy != "content":   # cf/hybrid: warm, sufficiently rated items only (whole ranking)
        full = rec.recommend_from_ratings(new_user_ratings, strategy=strategy, n=rec.n_items)
        assert 0 < len(full) < rec.n_items
        assert not movies.loc[full["movie_id"], "is_cold"].any()
        assert (rec.popularity["count"][movies.loc[full["movie_id"], "i"]] >= MIN_TRAIN_RATINGS).all()


def test_content_can_reach_cold_items(rec, new_user_ratings):
    df = rec.recommend_from_ratings(new_user_ratings, strategy="content", n=rec.n_items)
    cold_ids = set(rec.movies_table().query("is_cold")["movie_id"])
    assert cold_ids and cold_ids <= set(df["movie_id"])
    why = df.set_index("movie_id").loc[list(cold_ids), "why"]
    assert why.str.contains("New to the catalogue").all()
    rated_cold = sorted(cold_ids)[0]                                   # a rated cold movie is excluded
    df = rec.recommend_from_ratings({**new_user_ratings, rated_cold: 5}, strategy="content", n=rec.n_items)
    assert rated_cold not in set(df["movie_id"]) and (cold_ids - {rated_cold}) <= set(df["movie_id"])


def test_content_from_ratings_matches_brute_force(rec, new_user_ratings):
    from cinematch.content import profile_from_ratings, score_users_tt, user_vector_np

    items = {rec.item_index(m): float(r) for m, r in new_user_ratings.items()}
    scores = score_users_tt(rec.tt, user_vector_np(rec.tt, profile_from_ratings(items, rec.item_feats)))
    ok = (rec.warm_mask & (rec.popularity["count"] >= MIN_TRAIN_RATINGS)) | ~rec.warm_mask
    ok[list(items)] = False
    s = np.where(ok, scores, -np.inf)
    expected = np.lexsort((np.arange(len(s)), -s))[:N]
    got = rec.recommend_from_ratings(new_user_ratings, "content", N)["movie_id"].map(rec.item_index).to_numpy()
    assert np.array_equal(got, expected)


def test_recommend_from_ratings_edge_cases(rec):
    df = rec.recommend_from_ratings({}, strategy="hybrid", n=N)       # no ratings at all
    assert len(df) == N and df["movie_id"].is_unique
    df = rec.recommend_from_ratings({-1: 5, int(rec.movies_table()["movie_id"].iloc[0]): 9}, "cf", N)
    assert len(df) == N                                                 # unknown id ignored, rating clipped
    some_id = int(rec.movies_table()["movie_id"].iloc[1])
    for bad in (float("nan"), pd.NA, None):                             # NaN / NA / None ratings ignored
        for s in FROM_RATINGS_STRATEGIES:
            df = rec.recommend_from_ratings({some_id: bad}, s, N)
            assert len(df) == N and np.isfinite(df["score"]).all()
    assert rec.cluster_for_ratings({some_id: float("nan")}) == rec.largest_cluster()
    with pytest.raises(ValueError):
        rec.recommend_from_ratings({}, strategy="trending")


def test_cf_from_ratings_matches_brute_force(rec, new_user_ratings):
    from cinematch.cf import fold_in_user

    items = {rec.item_index(m): float(r) for m, r in new_user_ratings.items()}
    p, bu = fold_in_user(items, rec.cf, reg=1.0)
    scores = rec.cf["mu"] + bu + rec.cf["bi"] + rec.cf["Q"] @ p
    ok = rec.warm_mask & (rec.popularity["count"] >= MIN_TRAIN_RATINGS)
    ok[list(items)] = False
    s = np.where(ok, scores, -np.inf)
    expected = np.lexsort((np.arange(len(s)), -s))[:N]
    got = rec.recommend_from_ratings(new_user_ratings, "cf", N)["movie_id"].map(rec.item_index).to_numpy()
    assert np.array_equal(got, expected)


@pytest.mark.parametrize("strategy", USER_STRATEGIES)
def test_recommend_strategy_returns_n_distinct_warm_unseen(rec, strategy):
    for u in (0, 7, 42, rec.n_users - 1):
        items = rec.recommend_strategy(u, strategy, n=N)
        assert items.shape == (N,) and np.issubdtype(items.dtype, np.integer)
        assert len(np.unique(items)) == N
        assert rec.warm_mask[items].all()
        assert not np.isin(items, rec.user_train_items(u)).any()


def test_recommend_strategy_cf_is_masked_argmax(rec):
    u = 3
    scores = rec.cf["mu"] + rec.cf["bu"][u] + rec.cf["bi"] + rec.cf["Q"] @ rec.cf["P"][u]
    scores[~rec.warm_mask] = -np.inf
    scores[rec.user_train_items(u)] = -np.inf
    assert rec.recommend_strategy(u, "cf", 1)[0] == int(np.argmax(scores))
    with pytest.raises(ValueError):
        rec.recommend_strategy(u, "hybrid")


def _masked_top(rec, u, scores, n):
    s = np.where(rec.warm_mask, np.asarray(scores, dtype=np.float64), -np.inf)
    s[rec.user_train_items(u)] = -np.inf
    return np.lexsort((np.arange(len(s)), -s))[:n]


def test_recommend_strategy_content_and_trending_match_brute_force(rec):
    from cinematch.content import score_users_tt, user_vector_np

    for u in (1, 11, 99):
        tt_scores = score_users_tt(rec.tt, user_vector_np(rec.tt, rec.tt["user_feat_raw"][u]))
        assert np.array_equal(rec.recommend_strategy(u, "content", N), _masked_top(rec, u, tt_scores, N))
        trending, count = rec.popularity["trending"].astype(float), rec.popularity["count"].astype(float)
        s = np.where(rec.warm_mask, trending, -np.inf)
        s[rec.user_train_items(u)] = -np.inf
        expected = np.lexsort((np.arange(len(s)), -count, -s))[:N]     # trending, then count, then index
        assert np.array_equal(rec.recommend_strategy(u, "trending", N), expected)


def test_cluster_popular_is_filtered_stored_recs(rec):
    for u in (0, 5, 50):
        recs = np.asarray(rec.cl["recs"][str(rec.user_cluster(u))])
        expected = recs[~np.isin(recs, rec.user_train_items(u))][:N]
        assert np.array_equal(rec.recommend_strategy(u, "cluster_popular", N), expected)


def test_cluster_popular_tops_up_when_recs_exhausted(rec):
    from cinematch.clustering import cluster_recs

    u = 0
    items = rec.recommend_strategy(u, "cluster_popular", n=60)          # clusters.json stores 50 recs
    assert len(items) == 60 and len(np.unique(items)) == 60
    assert not np.isin(items, rec.user_train_items(u)).any() and rec.warm_mask[items].all()
    for c in range(rec.k):                     # display scores rank exactly like clustering.cluster_recs
        s = np.where(rec.warm_mask, rec._cluster_scores(c), -np.inf)
        expected = cluster_recs(rec.cl["centroids_raw"][c], rec.cf, rec.popularity, rec.warm_mask, k=60)
        assert np.array_equal(np.lexsort((np.arange(len(s)), -s))[:60], expected)


def test_recommend_for_cluster_and_clusters(rec, tiny_models):
    meta = json.loads((tiny_models["models_dir"] / "clusters.json").read_text(encoding="utf-8"))
    profiles = rec.clusters()
    assert len(profiles) == meta["k"] == rec.k
    assert {"cluster_id", "name", "size", "top_genres"} <= set(profiles[0])
    for c in profiles:
        df = rec.recommend_for_cluster(c["cluster_id"], n=N)
        assert len(df) == N and df["movie_id"].is_unique
        expected = rec.movies_table()["movie_id"].to_numpy()[meta["recs"][str(c["cluster_id"])][:N]]
        assert np.array_equal(df["movie_id"].to_numpy(), expected)
        assert df["why"].str.contains(c["name"], regex=False).all()
    df = rec.recommend_for_cluster(0, n=60)                             # past the 50 stored recs
    assert len(df) == 60 and df["movie_id"].is_unique
    assert not rec.movies_table().set_index("movie_id").loc[df["movie_id"], "is_cold"].any()
    with pytest.raises(ValueError):
        rec.recommend_for_cluster(rec.k)


def test_cluster_for_ratings_and_user_cluster(rec, new_user_ratings, tiny_models):
    from cinematch.cf import fold_in_user
    from cinematch.clustering import assign_cluster

    assert 0 <= rec.cluster_for_ratings(new_user_ratings) < rec.k
    assert rec.cluster_for_ratings({}) == rec.largest_cluster() == int(np.argmax(rec.cl["sizes"]))
    cold_ids = rec.movies_table().query("is_cold")["movie_id"].tolist()
    assert rec.cluster_for_ratings({m: 5 for m in cold_ids}) == rec.largest_cluster()
    rs = pd.read_parquet(tiny_models["processed_dir"] / "ratings_split.parquet")
    tr = rs[rs["split"] == "train"]
    movie_ids = rec.movies_table()["movie_id"].to_numpy()
    users = np.arange(0, rec.n_users, 4)
    assert 0 <= rec.user_cluster(0) < rec.k
    got_all = []
    for u in users:
        rows = tr[tr["u"] == u]
        ratings = dict(zip(movie_ids[rows["i"].to_numpy()].tolist(), rows["rating"].astype(float).tolist()))
        got = rec.cluster_for_ratings(ratings)
        warm = {int(i): float(r) for i, r in zip(rows["i"], rows["rating"]) if rec.warm_mask[i]}
        assert got == assign_cluster(rec.cl, fold_in_user(warm, rec.cf, reg=1.0)[0])
        got_all.append(got)
    # Not constant: real users land in several profiles. (Agreement with the stored label is ~78% on
    # real data; the tiny synthetic clusters are near-random, so it is not asserted here.)
    assert len(set(got_all)) > 1


def test_popular_for_rating(rec):
    df = rec.popular_for_rating(12)
    assert list(df.columns) == ["movie_id", "title", "year", "genres", "n_ratings"]
    assert len(df) == 12 and df["movie_id"].is_unique
    movies = rec.movies_table().set_index("movie_id")
    assert not movies.loc[df["movie_id"], "is_cold"].any()
    assert df["n_ratings"].max() == rec.popularity["count"][rec.warm_mask].max()


def test_user_train_items_and_movies_table(rec, tiny_models):
    rs = pd.read_parquet(tiny_models["processed_dir"] / "ratings_split.parquet")
    tr = rs[(rs["split"] == "train") & (rs["u"] == 5)]
    assert np.array_equal(np.sort(rec.user_train_items(5)), np.sort(tr["i"].to_numpy()))
    with pytest.raises(IndexError):
        rec.user_train_items(rec.n_users)
    m = rec.movies_table()
    assert np.array_equal(m["i"].to_numpy(), np.arange(rec.n_items))


def test_out_of_sync_artifacts_raise(tiny_models, tmp_path):
    models, processed = tmp_path / "models", tmp_path / "processed"
    shutil.copytree(tiny_models["models_dir"], models)
    shutil.copytree(tiny_models["processed_dir"], processed)
    with np.load(processed / "user_train_items.npz") as z:
        indptr, indices = z["indptr"], z["indices"]
    np.savez(processed / "user_train_items.npz", indptr=indptr[:-1], indices=indices)   # one user short
    with pytest.raises(ValueError, match="out of sync"):
        Recommender(models, processed)


def test_explain(rec):
    for s in FROM_RATINGS_STRATEGIES + USER_STRATEGIES:
        assert len(rec.explain(s)) > 20
    with pytest.raises(ValueError):
        rec.explain("magic")


def test_import_does_not_load_torch():
    code = "import cinematch.recommend, cinematch.bandit, sys; assert 'torch' not in sys.modules, 'torch loaded'"
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    res = subprocess.run([sys.executable, "-c", code], env=env, cwd=ROOT, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
