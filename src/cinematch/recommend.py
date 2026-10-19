"""Unified numpy-only inference for the app and the bandit (spec section 5, `recommend.py`).

`Recommender` loads the exported artifacts (`.npz` / `.json` / `movies.parquet`) and serves:

* new users from a handful of ratings (`recommend_from_ratings`): CF fold-in (closed-form ridge
  against frozen item vectors), the two-tower content model, or a hybrid of both;
* taste profiles (`recommend_for_cluster`, `cluster_for_ratings`);
* existing users by strategy (`recommend_strategy`), which the bandit uses to build its arm lists.

Only numpy and pandas are used: the helpers imported from `cf`, `content` and `clustering` are numpy
functions, and those modules import torch / sklearn lazily inside their training code only, so
`import cinematch.recommend` never loads torch (asserted in tests/test_recommend.py).

Candidate rules: CF, hybrid, cluster and bandit strategies only rank WARM items (cold items have
untrained CF vectors). The content strategy of `recommend_from_ratings` may also return cold items,
because the two-tower model scores items from metadata alone; that is its cold-start purpose.
New-user recommendations additionally skip warm items with fewer than `min_train_ratings` train
ratings, so a RMSE-tuned item bias on a handful of ratings cannot fill the list with obscure titles.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .cf import fold_in_user, load_baseline, load_cf, score_users
from .clustering import assign_cluster, cluster_recs, load_clusters
from .config import GENRES, MODELS_DIR, PROCESSED_DIR, TOP_K
from .content import load_two_tower, profile_from_ratings, score_users_tt, user_vector_np

FROM_RATINGS_STRATEGIES = ("cf", "content", "hybrid")
USER_STRATEGIES = ("cf", "content", "cluster_popular", "trending")   # == bandit.ARMS (spec 7.1)
MODEL_FILES = ("cf_mf.npz", "cf_baseline.npz", "two_tower.npz", "clusters.npz", "clusters.json",
               "popularity.npz")
PROCESSED_FILES = ("movies.parquet", "item_features.npz", "index_maps.json", "user_train_items.npz")
FOLD_IN_REG = 1.0          # ridge strength for new-user CF fold-in (cf.fold_in_user default)
MIN_TRAIN_RATINGS = 20     # warm items need this many train ratings to be recommended to new users
LIKED_THRESHOLD = 4.0      # a rating >= this counts as "liked" in the explanations

_EXPLANATIONS = {
    "cf": "Collaborative filtering (matrix factorisation): movies liked by people whose ratings look "
          "like yours. Your ratings are folded in against the frozen item vectors with a closed-form "
          "ridge regression, so no retraining is needed.",
    "content": "Two-tower content model: a user tower turns your rating profile (genre affinities, "
               "rating habits, preferred era) into a vector, an item tower does the same for each "
               "movie's genres, year and title. It also covers brand-new movies that have no ratings.",
    "hybrid": "Hybrid: the collaborative-filtering and content scores are standardised (z-scores over "
              "the candidate movies) and averaged, so a movie must look good to both models.",
    "cluster_popular": "Taste profile: the movies that score highest for your K-means taste profile "
                       "(the centroid of similar users' CF vectors, plus a small popularity bonus).",
    "trending": "Trending: the movies rated most often during the last 90 days of the training data.",
}
_EXPLANATIONS["cluster"] = _EXPLANATIONS["cluster_popular"]


class ModelsMissingError(FileNotFoundError):
    """Raised when model or processed-data artifacts are missing; `.missing` lists the paths."""

    def __init__(self, missing: list):
        self.missing = [str(p) for p in missing]
        super().__init__(
            "Missing CineMatch artifacts: " + ", ".join(self.missing)
            + ". Run: python scripts/download_data.py; python scripts/train_all.py; "
              "python scripts/evaluate.py")


def _load_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _movies_path(processed_dir: Path) -> Path | None:
    """movies.parquet, or the `.pkl` fallback written by data._save_df, or None."""
    for p in (processed_dir / "movies.parquet", processed_dir / "movies.pkl"):
        if p.exists():
            return p
    return None


def _top_n(scores: np.ndarray, n: int) -> np.ndarray:
    """Indices of the n largest finite scores, descending; ties broken by lower index."""
    order = np.lexsort((np.arange(len(scores)), -scores))
    order = order[np.isfinite(scores[order])]
    return order[:n].astype(np.int64)


def _zscore(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Standardise x over the masked entries (others become -inf)."""
    out = np.full(len(x), -np.inf)
    vals = x[mask].astype(np.float64)
    if len(vals):
        out[mask] = (vals - vals.mean()) / (vals.std() + 1e-12)
    return out


class Recommender:
    """Numpy-only recommender over the exported artifacts (spec section 5)."""

    def __init__(self, models_dir=MODELS_DIR, processed_dir=PROCESSED_DIR):
        models_dir, processed_dir = Path(models_dir), Path(processed_dir)
        missing = [models_dir / f for f in MODEL_FILES if not (models_dir / f).exists()]
        for f in PROCESSED_FILES:
            ok = _movies_path(processed_dir) is not None if f == "movies.parquet" else (processed_dir / f).exists()
            if not ok:
                missing.append(processed_dir / f)
        if missing:
            raise ModelsMissingError(missing)

        self.models_dir, self.processed_dir = models_dir, processed_dir
        self.cf = load_cf(models_dir)
        self.baseline = load_baseline(models_dir)
        self.tt = load_two_tower(models_dir)
        self.cl = load_clusters(models_dir)
        self.popularity = _load_npz(models_dir / "popularity.npz")
        self.item_feats = _load_npz(processed_dir / "item_features.npz")
        self.index_maps = json.loads((processed_dir / "index_maps.json").read_text(encoding="utf-8"))
        uti = _load_npz(processed_dir / "user_train_items.npz")
        self._uti_indptr, self._uti_indices = uti["indptr"], uti["indices"]

        mp = _movies_path(processed_dir)
        movies = pd.read_parquet(mp) if mp.suffix == ".parquet" else pd.read_pickle(mp)
        self._movies = movies.sort_values("i").reset_index(drop=True)
        self.n_items = len(self._movies)
        self.n_users = int(self.cf["P"].shape[0])
        if not np.array_equal(self._movies["i"].to_numpy(), np.arange(self.n_items)):
            raise ValueError("movies table item indices are not 0..I-1")
        self._movie_ids = self._movies["movie_id"].to_numpy(np.int64)
        self._id_to_idx = {int(m): k for k, m in enumerate(self._movie_ids)}
        self.warm_mask = ~np.asarray(self.item_feats["is_cold"], dtype=bool)
        self._train_count = np.asarray(self.popularity["count"], dtype=np.float64)
        self._genre = np.asarray(self.item_feats["genre_multi_hot"], dtype=bool)
        self._labels = np.asarray(self.cl["labels"], dtype=np.int64)
        self.k = int(self.cl["k"])
        self._check_consistent()

    def _check_consistent(self) -> None:
        """Fail early (not with an IndexError deep in a request) if models/ and processed/ differ."""
        n_items_ok = (self.cf["Q"].shape[0] == len(self.item_feats["is_cold"])
                      == self.tt["item_vectors"].shape[0] == len(self._train_count) == self.n_items)
        n_users_ok = (len(self._uti_indptr) == self.n_users + 1 == len(self._labels) + 1
                      == len(self.tt["user_feat_raw"]) + 1)
        if not (n_items_ok and n_users_ok):
            raise ValueError("CineMatch artifacts out of sync (models/ and data/processed/ come from "
                             "different runs); rerun python scripts/train_all.py")

    # ------------------------------------------------------------------ basic accessors
    def user_train_items(self, u: int) -> np.ndarray:
        """Train item indices of existing user index u (from user_train_items.npz)."""
        if not 0 <= u < self.n_users:
            raise IndexError(f"user index {u} out of range 0..{self.n_users - 1}")
        return self._uti_indices[self._uti_indptr[u]:self._uti_indptr[u + 1]].astype(np.int64)

    def movies_table(self) -> pd.DataFrame:
        """Copy of the movies table (movie_id, title, year, genres, i, is_cold, n_train_ratings, ...)."""
        return self._movies.copy()

    def item_index(self, movie_id: int) -> int:
        """Item index of a movie_id (KeyError if unknown)."""
        return self._id_to_idx[int(movie_id)]

    def user_cluster(self, u: int) -> int:
        """Taste-profile id of existing user index u (clusters.npz labels)."""
        return int(self._labels[u])

    def clusters(self) -> list[dict]:
        """Per cluster: {"cluster_id", "name", "size", "top_genres"} (top_genres as stored in clusters.json)."""
        return [{"cluster_id": c, "name": self.cl["names"][c], "size": int(self.cl["sizes"][c]),
                 "top_genres": [list(g) for g in self.cl["top_genres"][c]]} for c in range(self.k)]

    def explain(self, strategy: str) -> str:
        """One-paragraph plain-English description of a strategy."""
        if strategy not in _EXPLANATIONS:
            raise ValueError(f"unknown strategy {strategy!r}; expected one of {sorted(_EXPLANATIONS)}")
        return _EXPLANATIONS[strategy]

    def popular_for_rating(self, n: int = 30) -> pd.DataFrame:
        """Well-known warm movies for the rating widget, most-rated (train) first, with at most
        ceil(n/5) movies per primary genre (first listed genre) so the grid is not all one genre;
        if the cap leaves the grid short (tiny catalogues) it is filled in count order.
        Columns: movie_id, title, year, genres, n_ratings."""
        order = _top_n(np.where(self.warm_mask, self._train_count, -np.inf), self.n_items)
        primary = self._movies["genres"].str.split("|").str[0].to_numpy()
        cap, per_genre, picked = max(1, math.ceil(n / 5)), {}, []
        for j in order:
            if len(picked) == n:
                break
            if per_genre.get(primary[j], 0) < cap:
                per_genre[primary[j]] = per_genre.get(primary[j], 0) + 1
                picked.append(int(j))
        if len(picked) < n:
            taken = set(picked)
            picked += [int(j) for j in order if int(j) not in taken][:n - len(picked)]
        picked = np.array(sorted(picked, key=lambda j: (-self._train_count[j], j)), dtype=np.int64)
        out = self._frame(picked, self._train_count[picked], why=None).drop(columns="why")
        return out.rename(columns={"score": "n_ratings"}).astype({"n_ratings": np.int64})

    # ------------------------------------------------------------------ new users
    def _ratings_to_items(self, ratings: dict) -> dict[int, float]:
        """{movie_id: rating} -> {item index: rating}; unknown movie_ids and None / NaN ratings are
        ignored, ratings are clipped to 1..5."""
        out = {}
        for mid, r in ratings.items():
            idx = self._id_to_idx.get(int(mid))
            if idx is None or r is None or pd.isna(r) or not np.isfinite(float(r)):
                continue
            out[idx] = float(np.clip(float(r), 1.0, 5.0))
        return out

    def _fold_in(self, items: dict[int, float]) -> tuple[np.ndarray, float]:
        """CF vector + bias for a new user; cold items are skipped (their CF vectors are untrained)."""
        warm = {i: r for i, r in items.items() if self.warm_mask[i]}
        return fold_in_user(warm, self.cf, reg=FOLD_IN_REG)

    def _new_user_scores(self, items: dict[int, float], strategy: str) -> tuple[np.ndarray, np.ndarray]:
        """Scores [I] (-inf = not a candidate) and the raw content profile [40] for a new user."""
        rated = np.zeros(self.n_items, dtype=bool)
        rated[list(items)] = True
        warm_ok = self.warm_mask & (self._train_count >= MIN_TRAIN_RATINGS) & ~rated
        profile = (profile_from_ratings(items, self.item_feats) if items
                   else np.zeros(self.tt["feat_mean"].shape[0], dtype=np.float32))
        cf_scores = content_scores = None
        if strategy in ("cf", "hybrid"):
            p, bu = self._fold_in(items)
            cf_scores = score_users(self.cf, p, bu)[0].astype(np.float64)
        if strategy in ("content", "hybrid"):
            uv = user_vector_np(self.tt, profile)
            content_scores = np.asarray(score_users_tt(self.tt, uv), dtype=np.float64)
        if strategy == "cf":
            return np.where(warm_ok, cf_scores, -np.inf), profile
        if strategy == "content":   # cold items allowed: the two-tower model scores them from metadata
            return np.where(warm_ok | (~self.warm_mask & ~rated), content_scores, -np.inf), profile
        return (_zscore(cf_scores, warm_ok) + _zscore(content_scores, warm_ok)) / 2.0, profile

    def recommend_from_ratings(self, ratings: dict, strategy: str = "cf", n: int = TOP_K) -> pd.DataFrame:
        """Top-n unrated movies for a new user given {movie_id: rating}.

        strategy: "cf" (fold-in predicted rating), "content" (two-tower predicted rating; may include
        cold items) or "hybrid" (mean of z-scored cf and content scores over warm candidates).
        Columns: movie_id, title, year, genres, score, why.

        `score` scale depends on the strategy: for "cf" and "content" it is a predicted rating clipped
        to 1..5 (ranking uses the unclipped value, so displayed ties keep the model's order); for
        "hybrid" it is a relative z-score (0 = average candidate), not a rating; label it as such.
        """
        if strategy not in FROM_RATINGS_STRATEGIES:
            raise ValueError(f"unknown strategy {strategy!r}; expected one of {FROM_RATINGS_STRATEGIES}")
        items = self._ratings_to_items(ratings)
        scores, profile = self._new_user_scores(items, strategy)
        top = _top_n(scores, n)
        why = [self._why(strategy, int(j), items, profile) for j in top]
        shown = scores[top] if strategy == "hybrid" else np.clip(scores[top], 1.0, 5.0)
        return self._frame(top, shown, why)

    def largest_cluster(self) -> int:
        """Id of the taste profile with the most users (fallback when nothing is known)."""
        return int(np.argmax(self.cl["sizes"]))

    def cluster_for_ratings(self, ratings: dict) -> int:
        """Taste profile (cluster id) of a new user: nearest centroid (cosine) to the folded-in CF
        vector. With no usable warm rated movie the vector is zero, so the largest cluster is returned."""
        items = self._ratings_to_items(ratings)
        if not any(self.warm_mask[i] for i in items):
            return self.largest_cluster()
        p, _ = self._fold_in(items)
        return assign_cluster(self.cl, p)

    # ------------------------------------------------------------------ taste profiles
    def _cluster_scores(self, cluster_id: int) -> np.ndarray:
        """Display score of every item, the clustering.cluster_recs formula: mu + bi + centroid_raw @ Q.T
        + 0.1*log1p(count). Ranking always goes through cluster_recs itself (tests check they agree)."""
        c = np.asarray(self.cl["centroids_raw"][cluster_id], dtype=np.float64)
        return (self.cf["mu"] + self.cf["bi"].astype(np.float64) + self.cf["Q"].astype(np.float64) @ c
                + 0.1 * np.log1p(self._train_count))

    def _cluster_top(self, cid: int, cand: np.ndarray, n: int) -> np.ndarray:
        """Stored clusters.json recs of cluster cid that are candidates, topped up with
        clustering.cluster_recs over the remaining candidates when fewer than n survive."""
        recs = np.asarray(self.cl["recs"][str(cid)], dtype=np.int64)
        recs = recs[cand[recs]][:n]
        if len(recs) >= n:
            return recs
        rest_mask = cand.copy()
        rest_mask[recs] = False
        rest = cluster_recs(self.cl["centroids_raw"][cid], self.cf, self.popularity, rest_mask,
                            k=n - len(recs))
        return np.concatenate([recs, rest]).astype(np.int64)

    def recommend_for_cluster(self, cluster_id: int, n: int = TOP_K) -> pd.DataFrame:
        """Top-n warm movies of a taste profile: the precomputed clusters.json recs (50), best first,
        topped up with the same cluster score when n exceeds them."""
        cid = int(cluster_id)
        if not 0 <= cid < self.k:
            raise ValueError(f"cluster_id {cluster_id} out of range 0..{self.k - 1}")
        recs = self._cluster_top(cid, self.warm_mask, n)
        why = [f"Top pick for {self.cl['names'][cid]}"] * len(recs)
        return self._frame(recs, self._cluster_scores(cid)[recs], why)

    # ------------------------------------------------------------------ existing users (bandit arms)
    def recommend_strategy(self, u: int, strategy: str, n: int = TOP_K) -> np.ndarray:
        """Top-n item INDICES for existing user index u; candidates = warm items not in u's train set.

        cf: MF prediction; content: two-tower on the stored train profile (two_tower.npz user_feat_raw);
        cluster_popular: the user's cluster recs (clusters.json) filtered by candidates, topped up by the
        same cluster score if fewer than n survive; trending: last-90-days train counts (ties -> overall
        train count, then lower index).
        """
        if strategy not in USER_STRATEGIES:
            raise ValueError(f"unknown strategy {strategy!r}; expected one of {USER_STRATEGIES}")
        cand = self.warm_mask.copy()
        cand[self.user_train_items(u)] = False
        if strategy == "cf":
            scores = score_users(self.cf, self.cf["P"][u], self.cf["bu"][u])[0].astype(np.float64)
        elif strategy == "content":
            uv = user_vector_np(self.tt, self.tt["user_feat_raw"][u])
            scores = np.asarray(score_users_tt(self.tt, uv), dtype=np.float64)
        elif strategy == "trending":
            trending = np.asarray(self.popularity["trending"], dtype=np.float64)
            scores = trending + self._train_count / (self._train_count.max() + 1.0)   # tie-break term < 1
        else:
            return self._cluster_top(int(self._labels[u]), cand, n)
        return _top_n(np.where(cand, scores, -np.inf), n)

    # ------------------------------------------------------------------ output helpers
    def _frame(self, items: np.ndarray, scores: np.ndarray, why) -> pd.DataFrame:
        m = self._movies.iloc[items]
        return pd.DataFrame({"movie_id": m["movie_id"].to_numpy(), "title": m["title"].to_numpy(),
                             "year": m["year"].to_numpy(), "genres": m["genres"].to_numpy(),
                             "score": np.asarray(scores, dtype=np.float64),
                             "why": list(why) if why is not None else [""] * len(items)})

    def _because(self, j: int, items: dict[int, float]) -> str | None:
        """"Because you liked X": the liked warm movie whose CF item vector is closest (cosine) to j."""
        liked = [i for i, r in items.items() if r >= LIKED_THRESHOLD and self.warm_mask[i]]
        if not liked:
            return None
        Q = self.cf["Q"].astype(np.float64)
        L = Q[liked] / (np.linalg.norm(Q[liked], axis=1, keepdims=True) + 1e-12)
        best = liked[int(np.argmax(L @ (Q[j] / (np.linalg.norm(Q[j]) + 1e-12))))]
        return f"Because you liked {self._movies['title'].iloc[best]}"

    def _genre_match(self, j: int, profile: np.ndarray) -> str | None:
        """"Matches your taste for A & B": genres of j with positive affinity in the user profile."""
        affinity = np.asarray(profile[:18], dtype=np.float64)
        gs = sorted((g for g in np.flatnonzero(self._genre[j]) if affinity[g] > 0), key=lambda g: -affinity[g])
        return "Matches your taste for " + " & ".join(GENRES[g] for g in gs[:2]) if gs else None

    def _why(self, strategy: str, j: int, items: dict[int, float], profile: np.ndarray) -> str:
        parts = []
        if not self.warm_mask[j]:
            parts.append("New to the catalogue (no ratings yet), scored from its metadata")
        if strategy in ("cf", "hybrid"):
            parts.append(self._because(j, items) or "Liked by people who rate like you")
        if strategy in ("content", "hybrid"):
            parts.append(self._genre_match(j, profile) or "Similar in style and era to what you rated")
        return "; ".join(parts)
