"""Tests for cinematch.clustering on small synthetic data with planted taste clusters."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from cinematch import clustering
from cinematch.config import GENRES, Config

G = {g: j for j, g in enumerate(GENRES)}
PLANTED = [("Horror", "Thriller"), ("Comedy", "Romance"), ("Sci-Fi", "War")]
N_PER_CLUSTER, N_ITEMS, DIM = 60, 90, 8


def _synthetic(seed: int = 0, noise: float = 0.05) -> tuple[dict, dict, dict, np.ndarray]:
    """Planted world: 3 user clusters (tight directions in P), each loving the items of its 2 genres.

    Items 0..29 belong to cluster 0's genres, 30..59 to cluster 1's, 60..89 to cluster 2's; the last
    item of each block is cold. Extra users: index 180 is flagged, index 181 has no train rows.
    Returns (data, cf, popularity, true_cluster_of_fit_users).
    """
    rng = np.random.default_rng(seed)
    n_fit = 3 * N_PER_CLUSTER
    n_users = n_fit + 2
    truth = np.repeat(np.arange(3), N_PER_CLUSTER)
    dirs = np.linalg.qr(rng.normal(size=(DIM, 3)))[0].T                 # 3 orthonormal directions
    P = np.zeros((n_users, DIM))
    P[:n_fit] = dirs[truth] * rng.uniform(0.5, 1.5, size=(n_fit, 1)) + noise * rng.normal(size=(n_fit, DIM))
    P[n_fit:] = 0.05 * rng.normal(size=(2, DIM))
    item_block = np.repeat(np.arange(3), N_ITEMS // 3)
    Q = dirs[item_block] + 0.05 * rng.normal(size=(N_ITEMS, DIM))

    genre = np.zeros((N_ITEMS, len(GENRES)), dtype=np.float32)
    for i, b in enumerate(item_block):
        genre[i, [G[g] for g in PLANTED[b]]] = 1.0
        genre[i, G["Drama"]] = 1.0 if i % 2 else 0.0                      # shared background genre
    is_cold = np.zeros(N_ITEMS, dtype=bool)
    is_cold[[29, 59, 89]] = True

    rows = []
    for u in range(n_users - 1):                                         # last user has no train rows
        c = truth[u] if u < n_fit else rng.integers(3)
        for i in rng.choice(np.flatnonzero(~is_cold), size=40, replace=False):
            rating = 5.0 if item_block[i] == c else float(rng.integers(1, 3))
            rows.append((u, i, rating, "train"))
    rs = pd.DataFrame(rows, columns=["u", "i", "rating", "split"])
    rs["user_id"] = rs["u"] + 1
    rs["movie_id"] = rs["i"] + 1
    movies = pd.DataFrame({"movie_id": np.arange(N_ITEMS) + 1, "i": np.arange(N_ITEMS),
                           "genres": ["|".join(GENRES[j] for j in np.flatnonzero(g)) for g in genre]})
    data = {"ratings": rs, "movies": movies,
            "item_features": {"genre_multi_hot": genre, "is_cold": is_cold},
            "index_maps": {"user_ids": list(range(1, n_users + 1)), "flagged_user_ids": [n_fit + 1]}}
    cf = {"P": P.astype(np.float32), "Q": Q.astype(np.float32), "bu": np.zeros(n_users, np.float32),
          "bi": np.zeros(N_ITEMS, np.float32), "mu": np.float32(3.5)}
    popularity = {"score": np.ones(N_ITEMS, np.float32), "count": np.full(N_ITEMS, 10.0, np.float32),
                  "trending": np.ones(N_ITEMS, np.float32)}
    return data, cf, popularity, truth


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """Run train_step + eval_step once on the synthetic world (shared by several tests)."""
    root = tmp_path_factory.mktemp("clu")
    models_dir, figures_dir = root / "models", root / "figures"
    models_dir.mkdir()
    data, cf, popularity, truth = _synthetic()
    np.savez(models_dir / "cf_mf.npz", **cf)
    np.savez(models_dir / "popularity.npz", **popularity)
    cfg = Config(fast=True)                                              # k_max = 5
    cfg.k_min = 2                                                        # let the scan find the planted k=3
    ctx = {"data": data, "models_dir": models_dir, "figures_dir": figures_dir}
    train_out = clustering.train_step(cfg, ctx)
    eval_out = clustering.eval_step(cfg, {"models_dir": models_dir, "figures_dir": figures_dir})
    return {"train": train_out, "eval": eval_out, "models_dir": models_dir, "figures_dir": figures_dir,
            "data": data, "cf": cf, "truth": truth}


# ----------------------------------------------------------------------------- choose_k
def _scan(sil, min_frac=None, ks=None):
    ks = ks or list(range(2, 2 + len(sil)))
    return {"ks": ks, "inertia": [1.0] * len(ks), "silhouette": sil, "min_frac": min_frac or [0.1] * len(ks)}


def test_choose_k_picks_best_silhouette():
    assert clustering.choose_k(_scan([0.10, 0.30, 0.15, 0.12]), k_cap=8, min_frac=0.03) == 3


def test_choose_k_no_tie_band():
    # k=2 beats k=5 by only 0.015: no tie band any more (spec 17.2), so the best silhouette wins.
    scan = _scan([0.300, 0.200, 0.250, 0.285, 0.270])
    assert clustering.choose_k(scan, k_cap=8, min_frac=0.03) == 2


def test_choose_k_exact_tie_prefers_smaller_k():
    scan = _scan([0.05, 0.08, 0.08, 0.08], ks=[4, 5, 6, 7])
    assert clustering.choose_k(scan, k_cap=8, min_frac=0.03) == 5


def test_choose_k_respects_cap():
    sil = [0.05, 0.06, 0.07, 0.08, 0.09, 0.5, 0.6]                        # best at k=9, 10 (above cap)
    assert clustering.choose_k(_scan(sil, ks=list(range(4, 11))), k_cap=8, min_frac=0.03) == 8


def test_choose_k_respects_min_frac():
    scan = _scan([0.1, 0.5, 0.2], min_frac=[0.2, 0.01, 0.1], ks=[4, 5, 6])   # k=5 has a tiny cluster
    assert clustering.choose_k(scan, k_cap=8, min_frac=0.03) == 6


def test_choose_k_fallback_when_none_qualify():
    scan = _scan([0.4, 0.5, 0.6], min_frac=[0.01, 0.01, 0.02], ks=[5, 6, 7])
    assert clustering.choose_k(scan, k_cap=8, min_frac=0.03) == 4


def test_fit_kmeans_range_shapes():
    _, cf, _, _ = _synthetic()
    X = clustering._l2_normalise(cf["P"][:180])
    scan = clustering.fit_kmeans_range(X, 2, 5, seed=42)
    assert scan["ks"] == [2, 3, 4, 5]
    assert all(len(scan[key]) == 4 for key in ("inertia", "silhouette", "min_frac"))
    assert np.all(np.diff(scan["inertia"]) < 0)
    assert int(np.argmax(scan["silhouette"])) == 1                       # planted k = 3
    json.dumps(scan)                                                      # JSON-ready


# ----------------------------------------------------------------------------- naming
def test_name_clusters_recovers_planted_genres():
    data, _, _, truth = _synthetic()
    labels = np.concatenate([truth, [0, 1]])
    info = clustering.name_clusters(labels, data["ratings"], data["movies"],
                                    data["item_features"]["genre_multi_hot"])
    assert [d["cluster_id"] for d in info] == [0, 1, 2]
    # Both planted genres tag every liked movie of the cluster -> equal scores -> ML-1M genre order.
    assert [d["name"] for d in info] == ["Horror & Thriller fans", "Comedy & Romance fans", "Sci-Fi & War fans"]
    for d, genres in zip(info, PLANTED):
        assert {g for g, _ in d["top_genres"][:2]} == set(genres)
        assert d["top_genres"][0][1] > 50                                # ~ 100% vs ~ 33% globally
        assert abs(dict(d["top_genres"]).get("Drama", 0.0)) < 5          # background genre is not named
        assert len(d["top_genres"]) == 5
    assert [d["size"] for d in info] == [61, 61, 60]


def test_name_clusters_percentage_points_by_hand():
    # 4 items: 0 Comedy, 1 Comedy|Drama, 2 Drama, 3 Horror. Cluster 0 likes {0, 1}, cluster 1 likes {2, 3}.
    # Low ratings and non-train rows must be ignored.
    genre = np.zeros((4, len(GENRES)), dtype=np.float32)
    genre[0, G["Comedy"]] = genre[1, [G["Comedy"], G["Drama"]]] = genre[2, G["Drama"]] = genre[3, G["Horror"]] = 1
    rows = [(0, 0, 5.0, "train"), (0, 1, 4.0, "train"), (1, 2, 4.0, "train"), (1, 3, 5.0, "train"),
            (0, 3, 2.0, "train"), (1, 0, 5.0, "test"), (1, 1, 5.0, "val")]
    rs = pd.DataFrame(rows, columns=["u", "i", "rating", "split"])
    info = clustering.name_clusters(np.array([0, 1]), rs, None, genre)
    # Globally: Comedy 2/4, Drama 2/4, Horror 1/4. Cluster 0: Comedy 2/2, Drama 1/2 -> +50, 0, Horror -25.
    assert info[0]["top_genres"][0] == ["Comedy", 50.0]
    assert "Horror" not in dict(info[0]["top_genres"])                    # -25 pp: bottom, not top 5
    assert info[0]["name"] == "Comedy & Action fans"                      # 2nd = first 0.0 genre in order
    assert info[1]["top_genres"][0] == ["Horror", 25.0]
    assert info[1]["top_genres"][1] == ["Action", 0.0]


def test_name_clusters_deterministic_and_unique():
    data, _, _, truth = _synthetic()
    # Split planted cluster 0 into two halves: same taste -> same base name -> must be deduped.
    labels = np.concatenate([truth, [0, 1]]).copy()
    labels[:30] = 3
    args = (labels, data["ratings"], data["movies"], data["item_features"]["genre_multi_hot"])
    a = clustering.name_clusters(*args)
    b = clustering.name_clusters(*args)
    names = [d["name"] for d in a]
    assert names == [d["name"] for d in b]
    assert len(set(names)) == len(names) == 4
    assert names[0].startswith("Horror, Thriller & ") and names[3].startswith("Horror, Thriller & ")


def test_name_clusters_builds_genres_from_movies_when_missing():
    data, _, _, truth = _synthetic()
    labels = np.concatenate([truth, [0, 1]])
    a = clustering.name_clusters(labels, data["ratings"], data["movies"], None)
    b = clustering.name_clusters(labels, data["ratings"], data["movies"], data["item_features"]["genre_multi_hot"])
    assert a == b


def test_mainstream_name_when_no_over_representation():
    data, _, _, _ = _synthetic()
    labels = np.zeros(182, dtype=int)                                    # one cluster == everyone
    (d,) = clustering.name_clusters(labels, data["ratings"], data["movies"],
                                    data["item_features"]["genre_multi_hot"])
    assert d["name"] == "Mainstream"
    assert all(score == 0.0 for _, score in d["top_genres"])


def test_mainstream_threshold_and_duplicate_mainstream():
    # 100 items: 0..49 Comedy, 50..99 Drama. Clusters 0 and 1 (10 users each) are near the global mix
    # (+0.6 / +1.1 pp Drama: global Comedy share 50.6%), cluster 2 (1 user) strongly prefers Comedy,
    # cluster 3 (1 user) is just above the threshold (+2.7 pp Drama) and must get a genre name.
    # The two "Mainstream" clusters get "#id" suffixes.
    genre = np.zeros((100, len(GENRES)), dtype=np.float32)
    genre[:50, G["Comedy"]] = 1.0
    genre[50:, G["Drama"]] = 1.0
    liked = {0: list(range(0, 50)) + list(range(50, 100)),               # 50% Comedy
             1: list(range(0, 49)) + list(range(50, 100)),               # 49.5% Comedy
             2: list(range(0, 50)) + list(range(50, 60)),                # 83% Comedy
             3: list(range(0, 46)) + list(range(50, 100))}               # 52.1% Drama
    labels = np.array([0] * 10 + [1] * 10 + [2, 3])
    rows = [(u, i, 5.0, "train") for u in range(len(labels)) for i in liked[int(labels[u])]]
    info = clustering.name_clusters(labels, pd.DataFrame(rows, columns=["u", "i", "rating", "split"]), None, genre)
    assert [d["name"] for d in info] == ["Mainstream #0", "Mainstream #1", "Comedy & Action fans", "Drama & Action fans"]
    assert info[2]["top_genres"][0][0] == "Comedy" and info[2]["top_genres"][0][1] >= 2.0
    assert info[1]["top_genres"][0] == ["Drama", 1.1]                    # below 2.0 pp -> Mainstream
    assert info[3]["top_genres"][0] == ["Drama", 2.7]                    # above 2.0 pp -> genre name


def test_top_genres_never_negative_zero():
    # Cluster 0 is at -0.04 pp on Comedy: round(-0.04, 1) is -0.0, which must be stored as 0.0.
    genre = np.zeros((2000, len(GENRES)), dtype=np.float32)
    genre[:1000, G["Comedy"]] = 1.0
    labels = np.array([0, 1])
    rows = ([(0, i, 5.0, "train") for i in range(0, 999)] + [(0, i, 5.0, "train") for i in range(1000, 2000)]
            + [(1, i, 5.0, "train") for i in range(0, 1000)] + [(1, i, 5.0, "train") for i in range(1000, 1999)])
    info = clustering.name_clusters(labels, pd.DataFrame(rows, columns=["u", "i", "rating", "split"]), None, genre)
    for d in info:
        for _, sc in d["top_genres"]:
            assert not (sc == 0.0 and np.signbit(sc))
    assert dict(info[0]["top_genres"]).get("Comedy", 0.0) == 0.0


# ----------------------------------------------------------------------------- recs / assignment
def test_cluster_recs_excludes_cold_and_follows_taste():
    data, cf, popularity, _ = _synthetic()
    warm = ~data["item_features"]["is_cold"]
    centroid = cf["P"][:60].mean(axis=0)                                  # cluster-0 taste
    recs = clustering.cluster_recs(centroid, cf, popularity, warm, k=20)
    assert len(recs) == 20 and len(set(recs.tolist())) == 20
    assert warm[recs].all()
    assert np.all(recs < 30)                                              # cluster 0's item block


def test_assign_cluster_cosine():
    cl = {"centroids": np.array([[1.0, 0.0], [0.0, 1.0]])}
    assert clustering.assign_cluster(cl, np.array([5.0, 1.0])) == 0
    assert clustering.assign_cluster(cl, np.array([0.1, 0.3])) == 1


# ----------------------------------------------------------------------------- train / eval steps
def test_train_step_artifacts(trained):
    out, models_dir = trained["train"], trained["models_dir"]
    assert out["k"] == 3 and out["silhouette"] > 0.5
    assert len(out["names"]) == len(set(out["names"])) == 3
    cl = clustering.load_clusters(models_dir)
    U, d = trained["cf"]["P"].shape
    assert cl["centroids"].shape == (3, d) and cl["centroids_raw"].shape == (3, d)
    assert cl["labels"].shape == (U,) and cl["labels"].dtype == np.int32
    assert cl["labels"].min() >= 0 and cl["labels"].max() == 2
    assert cl["pca_2d"].shape == (U, 2) and cl["pca_components"].shape == (2, d) and cl["pca_mean"].shape == (d,)
    assert sum(cl["sizes"]) == U and cl["n_fit_users"] == 180
    assert set(cl["recs"]) == {"0", "1", "2"}
    warm = ~trained["data"]["item_features"]["is_cold"]
    for recs in cl["recs"].values():
        assert len(recs) == 50 and len(set(recs)) == 50 and warm[recs].all()
    for key in ("ks", "inertia", "silhouette", "min_frac"):
        assert len(cl["scan"][key]) == 4
    # Planted clusters recovered exactly (up to label permutation), and names match tastes.
    lab = cl["labels"][:180]
    truth = trained["truth"]
    for c in range(3):
        assert len(np.unique(lab[truth == c])) == 1
        name = cl["names"][lab[truth == c][0]]
        assert all(g in name for g in PLANTED[c])


def test_assign_cluster_matches_training_labels(trained):
    cl = clustering.load_clusters(trained["models_dir"])
    P = trained["cf"]["P"]
    a = np.array([clustering.assign_cluster(cl, p) for p in P])
    assert (a == cl["labels"]).all()


def test_assign_cluster_matches_labels_on_noisy_data(tmp_path):
    """Less separable world (noise 0.5): boundary users exist, and every stored label (fit users
    included) must still equal assign_cluster, i.e. one assignment rule for all users."""
    data, cf, popularity, _ = _synthetic(seed=1, noise=0.5)
    np.savez(tmp_path / "cf_mf.npz", **cf)
    np.savez(tmp_path / "popularity.npz", **popularity)
    cfg = Config(fast=True)
    out = clustering.train_step(cfg, {"data": data, "models_dir": tmp_path})
    cl = clustering.load_clusters(tmp_path)
    assert cfg.k_min == 4 and cl["scan"]["ks"][0] == cfg.k_min and out["k"] >= cfg.k_min
    a = np.array([clustering.assign_cluster(cl, p) for p in cf["P"]])
    assert (a == cl["labels"]).all()
    assert sum(out["sizes"]) == len(cf["P"])
    assert list(np.bincount(cl["labels"], minlength=out["k"])) == out["sizes"]


def test_train_step_names_ignore_flagged_user(tmp_path):
    """Names describe the non-flagged fit users only: a flagged user who loves a genre nobody else
    likes (Western) must not put it into any cluster's top genres."""
    data, cf, popularity, truth = _synthetic()
    flagged_u = 180                                                      # user_id 181 is flagged
    data["item_features"]["genre_multi_hot"][0, G["Western"]] = 1.0
    rs = data["ratings"]
    rs = rs[(rs["i"] != 0) | (rs["u"] == flagged_u)]
    extra = pd.DataFrame({"u": flagged_u, "i": 0, "rating": 5.0, "split": "train"}, index=range(2000))
    rs = pd.concat([rs, extra], ignore_index=True)
    rs["user_id"], rs["movie_id"] = rs["u"] + 1, rs["i"] + 1
    data["ratings"] = rs
    # Control: with the flagged user's rows included, Western enters its cluster's top genres.
    leaky = clustering.name_clusters(np.concatenate([truth, [0, 1]]), rs, None,
                                     data["item_features"]["genre_multi_hot"])
    assert dict(leaky[0]["top_genres"]).get("Western", 0.0) > 10      # item 0 is also Horror|Thriller

    np.savez(tmp_path / "cf_mf.npz", **cf)
    np.savez(tmp_path / "popularity.npz", **popularity)
    cfg = Config(fast=True)
    cfg.k_min = 2
    clustering.train_step(cfg, {"data": data, "models_dir": tmp_path})
    cl = clustering.load_clusters(tmp_path)
    assert cl["n_fit_users"] == 180
    for top in cl["top_genres"]:
        assert "Western" not in dict(top)
        assert dict(top).get("Western", 0.0) == 0.0


def test_eval_step_figures_and_metrics(trained):
    out, fig = trained["eval"], trained["figures_dir"]
    for name in ("elbow.png", "silhouette.png", "clusters_pca.png"):
        assert (fig / name).stat().st_size > 1000
    assert out == trained["train"]
    assert set(out) == {"k", "silhouette", "names", "sizes"}
    assert out["silhouette"] == round(out["silhouette"], 4)


def test_eval_step_missing_artifacts_returns_none(tmp_path):
    assert clustering.eval_step(Config(fast=True), {"models_dir": tmp_path, "figures_dir": tmp_path}) is None
