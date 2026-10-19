"""Unified numpy-only inference for the app and the bandit (spec section 5, `recommend.py`).

`Recommender` loads the exported artifacts (`.npz` / `.json` / `movies.parquet`) and serves:

* new users from a handful of ratings (`recommend_from_ratings`), with three strategies:
  - "knn": item-item cosine kNN (`item_knn.npz`) summed over the movies the user liked, plus a
    popularity prior knn_gamma * log1p(count);
  - "cf": matrix-factorisation fold-in (closed-form ridge against frozen item vectors) ranked by
    `cf.rank_scores` = predicted rating + beta * log1p(count);
  - "content": the two-tower content model + CONTENT_POP_WEIGHT * log1p(count) on warm items;
  beta and knn_gamma were chosen on the validation split by `cf.train_step` (`cf_ranking.json`);
* taste profiles (`recommend_for_cluster`, `cluster_for_ratings`);
* existing users by strategy (`recommend_strategy`), which the bandit uses to build its arm lists.

Why the popularity priors: ranking by MF predicted rating alone collapses to the item bias, which
rewards films rated 5 by a handful of people, so every user got the same "critics' list".

Only numpy, scipy.sparse and pandas are used: the helpers imported from `cf`, `content` and
`clustering` are numpy functions, and those modules import torch / sklearn lazily inside their
training code only, so `import cinematch.recommend` never loads torch (asserted in
tests/test_recommend.py).

Candidate rules: kNN, CF, cluster and bandit strategies only rank WARM items (cold items have no
train ratings, hence untrained CF vectors and empty kNN rows). The content strategy of
`recommend_from_ratings` may also return cold items, because the two-tower model scores items from
metadata alone; that is its cold-start purpose. New-user recommendations additionally skip warm
items with fewer than cfg.new_user_min_train_ratings train ratings.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .cf import (NEW_USER_FOLD_IN_REG, fold_in_user, knn_top_contributor, load_baseline, load_cf,
                 load_item_knn, new_user_knn_scores, rank_scores, seed_liked_items)
from .clustering import assign_cluster, cluster_recs, cluster_scores, load_clusters
from .config import GENRES, MODELS_DIR, PROCESSED_DIR, TOP_K, get_config
from .content import load_two_tower, profile_from_ratings, score_users_tt, user_vector_np

FROM_RATINGS_STRATEGIES = ("knn", "cf", "content")
USER_STRATEGIES = ("cf", "content", "cluster_popular", "trending")   # == bandit.ARMS (spec 7.1)
MODEL_FILES = ("cf_mf.npz", "cf_baseline.npz", "two_tower.npz", "clusters.npz", "clusters.json",
               "popularity.npz", "item_knn.npz", "cf_ranking.json")
PROCESSED_FILES = ("movies.parquet", "item_features.npz", "index_maps.json", "user_train_items.npz")
LIKED_THRESHOLD = 4.0      # a rating >= this counts as "liked" in the CF explanations
CONTENT_POP_WEIGHT = 0.5   # content strategy: + this * log1p(count) on warm items
ANCHOR_MIN_COS = 0.5       # cf "Because you liked X" only when cosine(Q_x, Q_item) >= this
ANCHOR_TOL = 0.1           # liked movies within this cosine of the best anchor may rotate
CF_GENERIC_WHY = "Popular with people whose ratings resemble yours"
POPULAR_WHY = "Popular with MovieLens viewers"

_EXPLANATIONS = {
    "knn": "Item-to-item neighbours: for every movie you liked (4+ stars) we look up the movies most "
           "often liked by the same people (cosine similarity of their audiences) and add them up. "
           "'Because you liked X' names the movie that contributed most.",
    "cf": "Collaborative filtering (matrix factorisation): your ratings are folded in against the "
          "frozen item vectors with a closed-form ridge regression, and movies are ranked by the "
          "predicted rating plus a popularity bonus (its weight was chosen on validation data). With "
          "only a few ratings this mostly returns well-known films liked by people whose ratings "
          "resemble yours; the list becomes more personal as you rate more movies.",
    "content": "Two-tower content model: a user tower turns your rating profile (genre affinities, "
               "rating habits, preferred era) into a vector, an item tower does the same for each "
               "movie's genres, year and title, plus a small popularity bonus. It also covers "
               "brand-new movies that have no ratings.",
    "cluster_popular": "Taste profile: the movies that score highest for your K-means taste profile "
                       "(the centroid of similar users' CF vectors, weighted up so the profile matters, "
                       "plus a popularity bonus).",
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


class Recommender:
    """Numpy-only recommender over the exported artifacts (spec section 5).

    `cfg` (default `get_config()`) supplies the new-user candidate floor and the taste-profile
    weights; beta and knn_gamma come from models/cf_ranking.json (chosen on validation)."""

    def __init__(self, models_dir=MODELS_DIR, processed_dir=PROCESSED_DIR, cfg=None):
        models_dir, processed_dir = Path(models_dir), Path(processed_dir)
        missing = [models_dir / f for f in MODEL_FILES if not (models_dir / f).exists()]
        for f in PROCESSED_FILES:
            ok = _movies_path(processed_dir) is not None if f == "movies.parquet" else (processed_dir / f).exists()
            if not ok:
                missing.append(processed_dir / f)
        if missing:
            raise ModelsMissingError(missing)

        self.models_dir, self.processed_dir = models_dir, processed_dir
        self.cfg = cfg if cfg is not None else get_config()
        self.min_train_ratings = int(self.cfg.new_user_min_train_ratings)
        self.cf = load_cf(models_dir)
        self.item_knn = load_item_knn(models_dir)
        self.ranking = json.loads((models_dir / "cf_ranking.json").read_text(encoding="utf-8"))
        self.beta = float(self.ranking["beta"])
        self.knn_gamma = float(self.ranking["knn_gamma"])
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
        self._log_count = np.log1p(self._train_count)
        self._pop_score = np.asarray(self.popularity["score"], dtype=np.float64)
        self._genre = np.asarray(self.item_feats["genre_multi_hot"], dtype=bool)
        self._labels = np.asarray(self.cl["labels"], dtype=np.int64)
        self.k = int(self.cl["k"])
        self._check_consistent()

    def _check_consistent(self) -> None:
        """Fail early (not with an IndexError deep in a request) if models/ and processed/ differ."""
        n_items_ok = (self.cf["Q"].shape[0] == len(self.item_feats["is_cold"])
                      == self.tt["item_vectors"].shape[0] == len(self._train_count) == self.n_items
                      and self.item_knn.shape == (self.n_items, self.n_items))
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
        if strategy == "knn" and self.knn_gamma > 0:
            return (_EXPLANATIONS["knn"] + " A small bonus for well-known films is added (its weight "
                    "was chosen on validation data).")
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
        return fold_in_user(self._warm_only(items), self.cf, reg=NEW_USER_FOLD_IN_REG)

    def _warm_only(self, items: dict[int, float]) -> dict[int, float]:
        """The rated movies that are warm (cold movies have no train ratings: no CF vector, no kNN row)."""
        return {i: r for i, r in items.items() if self.warm_mask[i]}

    def _profile(self, items: dict[int, float]) -> np.ndarray:
        """Raw 40-dim content profile of a new user (zeros when nothing is rated)."""
        if items:
            return profile_from_ratings(items, self.item_feats)
        return np.zeros(self.tt["feat_mean"].shape[0], dtype=np.float32)

    def _new_user_scores(self, items: dict[int, float], strategy: str) -> np.ndarray:
        """Ranking score of every item [I] for a new user; -inf = not a candidate.

        Candidates: warm unrated items with >= min_train_ratings train ratings; the content strategy
        may also return unrated cold items."""
        rated = np.zeros(self.n_items, dtype=bool)
        rated[list(items)] = True
        warm_ok = self.warm_mask & (self._train_count >= self.min_train_ratings) & ~rated
        if strategy == "knn":   # seeds rated >= 4, else >= 3, else popularity (cf.new_user_knn_scores)
            scores = new_user_knn_scores(self.item_knn, self._warm_only(items), self._train_count,
                                         self.knn_gamma, self._pop_score)
        elif strategy == "cf":
            p, bu = self._fold_in(items)
            scores = rank_scores(self.cf, p, bu, self._train_count, self.beta)[0]
        else:   # content: cold items allowed, the two-tower model scores them from metadata
            uv = user_vector_np(self.tt, self._profile(items))
            prior = np.where(self.warm_mask, CONTENT_POP_WEIGHT * self._log_count, 0.0)
            scores = np.asarray(score_users_tt(self.tt, uv), dtype=np.float64) + prior
            warm_ok = warm_ok | (~self.warm_mask & ~rated)
        return np.where(warm_ok, np.asarray(scores, dtype=np.float64), -np.inf)

    def recommend_from_ratings(self, ratings: dict, strategy: str = "knn", n: int = TOP_K) -> pd.DataFrame:
        """Top-n unrated movies for a new user given {movie_id: rating}.

        strategy: "knn" (item-kNN on the liked movies + popularity prior), "cf" (fold-in predicted
        rating + beta * log1p(count)) or "content" (two-tower + popularity prior; may include cold
        items). Columns: movie_id, title, year, genres, score, why. `score` is the ranking score; it
        orders the list and is not a predicted rating, so do not display it as stars.
        """
        if strategy not in FROM_RATINGS_STRATEGIES:
            raise ValueError(f"unknown strategy {strategy!r}; expected one of {FROM_RATINGS_STRATEGIES}")
        items = self._ratings_to_items(ratings)
        scores = self._new_user_scores(items, strategy)
        top = _top_n(scores, n)
        if strategy == "knn":
            reasons = self._knn_why(top, items)
        elif strategy == "cf":
            reasons = self._because(top, items)
        else:
            profile = self._profile(items)
            reasons = [self._genre_match(int(j), profile) or "Similar in style and era to what you rated"
                       for j in top]
        why = [self._why(int(j), reason) for j, reason in zip(top, reasons)]
        return self._frame(top, scores[top], why)

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
        """Display score of every item: clustering.cluster_scores (mu + bi + cluster_taste_weight *
        Q @ centroid_raw + cluster_pop_weight * log1p(count)), the formula cluster_recs ranks by."""
        return cluster_scores(self.cl["centroids_raw"][cluster_id], self.cf, self._train_count,
                              self.cfg.cluster_taste_weight, self.cfg.cluster_pop_weight)

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
                            k=n - len(recs), taste_weight=self.cfg.cluster_taste_weight,
                            pop_weight=self.cfg.cluster_pop_weight)
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

        cf: MF prediction + beta * log1p(count) (`cf.rank_scores`, beta from cf_ranking.json);
        content: two-tower on the stored train profile (two_tower.npz user_feat_raw);
        cluster_popular: the user's cluster recs (clusters.json) filtered by candidates, topped up by the
        same cluster score if fewer than n survive; trending: last-90-days train counts (ties -> overall
        train count, then lower index).
        """
        if strategy not in USER_STRATEGIES:
            raise ValueError(f"unknown strategy {strategy!r}; expected one of {USER_STRATEGIES}")
        cand = self.warm_mask.copy()
        cand[self.user_train_items(u)] = False
        if strategy == "cf":
            scores = rank_scores(self.cf, self.cf["P"][u], self.cf["bu"][u], self._train_count, self.beta)[0]
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

    def _knn_why(self, top: np.ndarray, items: dict[int, float]) -> list[str]:
        """"Because you liked X" per kNN pick, X = the seed movie with the largest item-item similarity
        to it (the top contributor to its score); a generic reason when no seed is similar."""
        seeds = seed_liked_items(self._warm_only(items))
        out = []
        for j in top:
            k = knn_top_contributor(self.item_knn, seeds, int(j)) if len(seeds) else -1
            out.append(f"Because you liked {self._movies['title'].iloc[k]}" if k >= 0 else POPULAR_WHY)
        return out

    def _because(self, top: np.ndarray, items: dict[int, float]) -> list[str]:
        """CF reason per recommended item: "Because you liked X" when a liked warm movie X has a CF item
        vector with cosine >= ANCHOR_MIN_COS to it, else CF_GENERIC_WHY. Among the liked movies within
        ANCHOR_TOL of the closest one, the one used least so far on this list is picked, so near-ties
        do not repeat one anchor on every card."""
        liked = [i for i, r in items.items() if r >= LIKED_THRESHOLD and self.warm_mask[i]]
        if not liked:
            return [CF_GENERIC_WHY] * len(top)
        Q = self.cf["Q"].astype(np.float64)
        Qn = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-12)
        cos = Qn[np.asarray(top, dtype=np.int64)] @ Qn[liked].T            # [len(top), len(liked)]
        used, out = np.zeros(len(liked), dtype=np.int64), []
        for row in cos:
            if row.max() < ANCHOR_MIN_COS:
                out.append(CF_GENERIC_WHY)
                continue
            near = np.flatnonzero(row >= max(row.max() - ANCHOR_TOL, ANCHOR_MIN_COS))
            k = min(near, key=lambda c: (used[c], -row[c]))
            used[k] += 1
            out.append(f"Because you liked {self._movies['title'].iloc[liked[k]]}")
        return out

    def _genre_match(self, j: int, profile: np.ndarray) -> str | None:
        """"Matches your taste for A & B": the genres of j that are among the user's top-2 genres by
        positive affinity (profile[:18]); None when j has neither."""
        affinity = np.asarray(profile[:18], dtype=np.float64)
        top2 = [g for g in np.lexsort((np.arange(len(affinity)), -affinity))[:2] if affinity[g] > 0]
        gs = [g for g in top2 if self._genre[j, g]]
        return "Matches your taste for " + " & ".join(GENRES[g] for g in gs) if gs else None

    def _why(self, j: int, reason: str) -> str:
        """Card text: the strategy's reason, prefixed by a cold-start note for cold items."""
        if not self.warm_mask[j]:
            return "New to the catalogue (no ratings yet), scored from its metadata; " + reason
        return reason
