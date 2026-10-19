"""Tests for cinematch.data and scripts/download_data.py (synthetic data only, tmp dirs only)."""
import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cinematch import data
from cinematch.config import GENRES

ROOT = Path(__file__).resolve().parents[1]


# ----------------------------------------------------------------------------- raw + features
def test_load_raw_parses_dat_files(tiny_raw_dir, tiny_ratings):
    ratings, movies, users = data.load_raw(tiny_raw_dir)
    assert len(ratings) == len(tiny_ratings)
    assert ratings["rating"].dtype == np.float32
    assert list(ratings.columns) == ["user_id", "movie_id", "rating", "timestamp"]
    assert movies["year"].between(1990, 1999).all()
    assert len(users) == tiny_ratings["user_id"].nunique()


def test_load_raw_missing_dir_is_friendly(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_data"):
        data.load_raw(tmp_path / "nope")


def test_year_fallback_is_median():
    movies = pd.DataFrame({"movie_id": [1, 2, 3], "title": ["A (1990)", "B", "C (1994)"], "genres": ["Drama"] * 3})
    assert data._add_year(movies)["year"].tolist() == [1990, 1992, 1994]


def test_build_item_features_shapes(tiny_movies):
    feats = data.build_item_features(tiny_movies, n_svd=16, seed=42)
    n = len(tiny_movies)
    assert feats["genre_multi_hot"].shape == (n, len(GENRES)) and feats["genre_multi_hot"].dtype == np.float32
    assert feats["title_svd"].shape == (n, 16) and feats["title_svd"].dtype == np.float32
    assert feats["year_scaled"].shape == (n,) and feats["year_scaled"].min() >= 0 and feats["year_scaled"].max() <= 1
    assert feats["is_cold"].dtype == bool and not feats["is_cold"].any()
    n_genres = tiny_movies["genres"].str.split("|").str.len().to_numpy()
    np.testing.assert_array_equal(feats["genre_multi_hot"].sum(1), n_genres)


# ----------------------------------------------------------------------------- split invariants
@pytest.fixture
def split(tiny_ratings, tiny_movies, tiny_cfg):
    flagged = [1, 2]
    return data.split_ratings(tiny_ratings, tiny_movies, tiny_cfg, flagged)


def test_split_columns_and_sizes(split, tiny_ratings, tiny_cfg):
    rs, movies_out, maps = split
    assert list(rs.columns) == ["user_id", "movie_id", "rating", "timestamp", "u", "i", "split"]
    assert rs["u"].dtype == np.int32 and rs["i"].dtype == np.int32 and rs["rating"].dtype == np.float32
    assert len(rs) == len(tiny_ratings)
    assert set(rs["split"].unique()) <= {"train", "val", "test", "cold"}
    counts = tiny_ratings["movie_id"].value_counts()
    n_candidates = int((counts >= tiny_cfg.cold_item_min_ratings).sum())
    assert len(maps["cold_movie_ids"]) == round(tiny_cfg.cold_item_frac * n_candidates) > 0
    # per-user split sizes follow max(1, floor(0.1 n)) on warm ratings
    warm = rs[rs["split"] != "cold"]
    for _, g in warm.groupby("user_id"):
        n = len(g)
        k = max(1, int(np.floor(0.1 * n)))
        assert (g["split"] == "test").sum() == k
        assert (g["split"] == "val").sum() == k


def test_cold_items_only_in_cold_split(split):
    rs, movies_out, maps = split
    cold_ids = set(maps["cold_movie_ids"])
    assert set(rs.loc[rs["split"] == "cold", "movie_id"]) <= cold_ids
    assert not rs.loc[rs["split"] != "cold", "movie_id"].isin(cold_ids).any()
    assert set(movies_out.loc[movies_out["is_cold"], "movie_id"]) == cold_ids
    assert (movies_out.loc[movies_out["is_cold"], "n_train_ratings"] == 0).all()


def test_per_user_chronological_order(split):
    rs, _, _ = split
    warm = rs[rs["split"] != "cold"]
    order = {"train": 0, "val": 1, "test": 2}
    for _, g in warm.groupby("user_id"):
        g = g.sort_values(["timestamp", "movie_id"])
        codes = g["split"].map(order).to_numpy()
        assert np.all(np.diff(codes) >= 0), "train < val < test in time"


def test_every_user_has_test_row_and_flagged_kept(split, tiny_ratings):
    rs, _, maps = split
    assert set(rs.loc[rs["split"] == "test", "user_id"]) == set(tiny_ratings["user_id"])
    assert maps["flagged_user_ids"] == [1, 2]
    flagged_rows = rs[rs["user_id"].isin([1, 2])]
    assert {"train", "val", "test"} <= set(flagged_rows["split"])


def test_index_maps_bijective(split, tiny_movies):
    rs, movies_out, maps = split
    user_ids, movie_ids = np.array(maps["user_ids"]), np.array(maps["movie_ids"])
    assert np.all(np.diff(user_ids) > 0) and np.all(np.diff(movie_ids) > 0)
    assert len(movie_ids) == len(tiny_movies)
    np.testing.assert_array_equal(user_ids[rs["u"]], rs["user_id"])
    np.testing.assert_array_equal(movie_ids[rs["i"]], rs["movie_id"])
    np.testing.assert_array_equal(movies_out["i"], np.arange(len(movies_out)))
    np.testing.assert_array_equal(movies_out["movie_id"], movie_ids)


def test_split_is_deterministic(tiny_ratings, tiny_movies, tiny_cfg):
    a = data.split_ratings(tiny_ratings, tiny_movies, tiny_cfg, [])
    b = data.split_ratings(tiny_ratings, tiny_movies, tiny_cfg, [])
    pd.testing.assert_frame_equal(a[0], b[0])
    assert a[2] == b[2]


# ----------------------------------------------------------------------------- artifacts + helpers
def test_prepare_and_load_roundtrip(tiny_ratings, tiny_movies, tiny_cfg, tmp_processed_dir):
    out = data.prepare_data(tiny_cfg, [3], processed_dir=tmp_processed_dir, ratings=tiny_ratings, movies=tiny_movies)
    for name in ["ratings_split.parquet", "movies.parquet", "item_features.npz", "index_maps.json",
                 "split_stats.json", "user_train_items.npz"]:
        assert (tmp_processed_dir / name).exists(), name
    loaded = data.load_processed(tmp_processed_dir)
    assert set(loaded) == {"ratings", "movies", "item_features", "index_maps", "stats"}
    pd.testing.assert_frame_equal(loaded["ratings"], out["ratings"])
    pd.testing.assert_frame_equal(loaded["movies"], out["movies"])
    for k, v in out["item_features"].items():
        np.testing.assert_array_equal(loaded["item_features"][k], v)
    assert loaded["index_maps"] == out["index_maps"]
    stats = loaded["stats"]
    assert stats["n_train"] + stats["n_val"] + stats["n_test"] + stats["n_cold"] == len(tiny_ratings)
    assert stats["n_flagged"] == 1 and stats["n_items"] == len(tiny_movies)
    # CSR of train items matches user_train_items()
    with np.load(tmp_processed_dir / "user_train_items.npz") as z:
        indptr, indices = z["indptr"], z["indices"]
    assert indptr.dtype == np.int64 and indices.dtype == np.int32 and len(indptr) == stats["n_users"] + 1
    lists = data.user_train_items(out["ratings"], stats["n_users"])
    for u in (0, 7, stats["n_users"] - 1):
        np.testing.assert_array_equal(indices[indptr[u]:indptr[u + 1]], lists[u])


def test_load_processed_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="train_all"):
        data.load_processed(tmp_path)


def test_train_matrix_holdout_and_eval_users(split):
    rs, movies_out, maps = split
    n_u, n_i = len(maps["user_ids"]), len(maps["movie_ids"])
    R = data.train_matrix(rs, n_u, n_i)
    assert R.shape == (n_u, n_i) and R.nnz == int((rs["split"] == "train").sum())
    R_ex = data.train_matrix(rs, n_u, n_i, exclude_users=[0, 1])
    assert R_ex[0].nnz == 0 and R_ex[1].nnz == 0 and R_ex[2].nnz == R[2].nnz
    hold = data.holdout_dict(rs, "test")
    assert sum(len(v) for v in hold.values()) == int((rs["split"] == "test").sum())
    a = data.eval_users_sample(rs, 50, seed=42, exclude_users=[0, 1])
    b = data.eval_users_sample(rs, 50, seed=42, exclude_users=[0, 1])
    np.testing.assert_array_equal(a, b)
    assert len(a) == 50 and np.all(np.diff(a) > 0) and not np.isin([0, 1], a).any()
    assert len(data.eval_users_sample(rs, None, seed=42)) == n_u


def test_train_step_and_eval_step_use_ctx_dirs(tiny_cfg, tiny_ctx):
    tiny_ctx["flagged_user_ids"] = [1, 2]
    out = data.train_step(tiny_cfg, tiny_ctx)
    eval_users = np.load(tiny_ctx["models_dir"] / "eval_users.npy")
    assert len(eval_users) == tiny_cfg.eval_users == out["n_eval_users"]
    assert not np.isin([0, 1], eval_users).any(), "flagged users (indices 0, 1) excluded"
    assert out["n_flagged_users"] == 2
    expected_keys = {"n_users", "n_items", "n_cold_items", "n_flagged_users", "n_train", "n_val", "n_test",
                     "n_cold", "n_eval_users"}
    assert set(out) == expected_keys
    json.loads((tiny_ctx["processed_dir"] / "split_stats.json").read_text())

    ctx2 = {k: tiny_ctx[k] for k in ("processed_dir", "models_dir")}
    out2 = data.eval_step(tiny_cfg, ctx2)
    assert out2 == out
    d = ctx2["data"]
    for key in ("eval_users", "train_items", "holdout_test", "holdout_cold", "warm_mask"):
        assert key in d
    np.testing.assert_array_equal(d["eval_users"], eval_users)
    assert d["warm_mask"].sum() == out["n_items"] - out["n_cold_items"]


def test_eval_step_returns_none_without_artifacts(tiny_cfg, tmp_path):
    assert data.eval_step(tiny_cfg, {"processed_dir": tmp_path / "p", "models_dir": tmp_path / "m"}) is None


# ----------------------------------------------------------------------------- download script
def _load_download_module(monkeypatch):
    """Import scripts/download_data.py with sys.path / sys.modules changes undone after the test."""
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    if "_bootstrap" not in sys.modules:  # imported by download_data; drop it again at teardown
        monkeypatch.setitem(sys.modules, "_bootstrap", None)
        monkeypatch.delitem(sys.modules, "_bootstrap")
    spec = importlib.util.spec_from_file_location("download_data", ROOT / "scripts" / "download_data.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fake_zip_bytes(tmp_path) -> bytes:
    path = tmp_path / "src.zip"
    with zipfile.ZipFile(path, "w") as zf:
        for name in ("ratings.dat", "movies.dat", "users.dat"):
            zf.writestr(f"ml-1m/{name}", f"{name} " * 5000)
    return path.read_bytes()


def test_download_resumes_after_drop_and_extracts(tmp_path, monkeypatch):
    import requests

    mod = _load_download_module(monkeypatch)
    payload = _fake_zip_bytes(tmp_path)
    total = len(payload)
    calls = []

    class FakeResponse:
        def __init__(self, start, drop_after=None):
            self.start, self.drop_after = start, drop_after
            self.status_code = 206 if start else 200
            self.headers = {"Content-Length": str(total - start)}

        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size):
            body = payload[self.start:]
            sent = 0
            for k in range(0, len(body), 1000):
                if self.drop_after is not None and sent >= self.drop_after:
                    raise requests.ConnectionError("connection dropped")
                yield body[k:k + 1000]
                sent += 1000

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    class FakeSession:
        def get(self, url, headers=None, stream=True, timeout=None):
            start = int(headers["Range"].split("=")[1].rstrip("-")) if headers and "Range" in headers else 0
            calls.append(start)
            first_body_request = len(calls) == 2
            return FakeResponse(start, drop_after=3000 if first_body_request else None)

    monkeypatch.setattr(mod, "_session", lambda: FakeSession())
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)
    dest = tmp_path / "out" / "ml-1m.zip"
    mod.download_ml1m(dest_zip=dest, url="https://example.invalid/ml-1m.zip")
    assert dest.read_bytes() == payload
    assert calls[0] == 0 and calls[1] == 0 and calls[2] == 3000, "second body request resumes with Range"
    raw = mod.extract(dest, tmp_path / "raw")
    assert sorted(p.name for p in raw.iterdir()) == ["movies.dat", "ratings.dat", "users.dat"]
    # idempotent: a second call neither downloads nor extracts again
    mod.download_ml1m(dest_zip=dest, url="https://example.invalid/ml-1m.zip")
    assert len(calls) == 3
