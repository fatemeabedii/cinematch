"""MovieLens-1M loading, cold-item holdout, per-user time split, index maps and processed artifacts.

Pipeline steps 3-5 of the spec (section 4.3). numpy/pandas/scipy/sklearn only (never torch).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .config import GENRES, MODELS_DIR, PROCESSED_DIR, RAW_DIR, SEED

SPLITS = ("train", "val", "test", "cold")
_YEAR_RE = re.compile(r"\((\d{4})\)\s*$")


# ----------------------------------------------------------------------------- raw loading
def _read_dat(path: Path, names: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python scripts/download_data.py")
    return pd.read_csv(path, sep="::", engine="python", names=names, encoding="latin-1")


def _add_year(movies: pd.DataFrame) -> pd.DataFrame:
    """Add an int `year` column parsed from the title (fallback: median year)."""
    movies = movies.copy()
    years = movies["title"].astype(str).str.extract(_YEAR_RE, expand=False).astype(float)
    median = float(np.nanmedian(years.to_numpy())) if years.notna().any() else 1990.0
    movies["year"] = years.fillna(round(median)).astype(int)
    return movies


def load_raw(raw_dir: Path = RAW_DIR) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read ratings.dat, movies.dat, users.dat. Returns (ratings, movies, users).

    ratings: user_id, movie_id, rating (float32), timestamp. movies: movie_id, title, genres, year.
    users is loaded for completeness/EDA only (never a model feature).
    """
    raw_dir = Path(raw_dir)
    ratings = _read_dat(raw_dir / "ratings.dat", ["user_id", "movie_id", "rating", "timestamp"])
    ratings["rating"] = ratings["rating"].astype(np.float32)
    movies = _add_year(_read_dat(raw_dir / "movies.dat", ["movie_id", "title", "genres"]))
    users = _read_dat(raw_dir / "users.dat", ["user_id", "gender", "age", "occupation", "zip"])
    return ratings, movies, users


# ----------------------------------------------------------------------------- item features
def genre_multi_hot(genres: pd.Series) -> np.ndarray:
    """float32 [n, 18] multi-hot of pipe-separated genre strings, in config.GENRES order."""
    col = {g: j for j, g in enumerate(GENRES)}
    out = np.zeros((len(genres), len(GENRES)), dtype=np.float32)
    for row, s in enumerate(genres.astype(str).to_numpy()):
        for g in s.split("|"):
            if g in col:
                out[row, col[g]] = 1.0
    return out


def build_item_features(movies: pd.DataFrame, n_svd: int = 16, seed: int = SEED) -> dict[str, np.ndarray]:
    """Item features aligned with the row order of `movies` (expected sorted by movie_id / item index).

    Keys: genre_multi_hot [I,18], year_scaled [I] ((year-1919)/81 clipped to 0..1),
    title_svd [I,n_svd] (TF-IDF of lowercase title without year -> TruncatedSVD), is_cold bool [I].
    The SVD is fit on all titles incl. cold items (unsupervised metadata, no ratings).
    """
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    if "year" not in movies.columns:
        movies = _add_year(movies)
    titles = [_YEAR_RE.sub("", t).strip().lower() for t in movies["title"].astype(str)]
    tfidf = TfidfVectorizer().fit_transform(titles)
    n_comp = max(1, min(n_svd, tfidf.shape[1] - 1, len(titles) - 1))
    svd = TruncatedSVD(n_components=n_comp, random_state=seed).fit_transform(tfidf)
    title_svd = np.zeros((len(titles), n_svd), dtype=np.float32)
    title_svd[:, :n_comp] = svd  # zero-padded if the vocabulary is tiny
    year = movies["year"].to_numpy(dtype=np.float64)
    is_cold = (movies["is_cold"].to_numpy(dtype=bool) if "is_cold" in movies.columns
               else np.zeros(len(movies), dtype=bool))
    return {
        "genre_multi_hot": genre_multi_hot(movies["genres"]),
        "year_scaled": np.clip((year - 1919) / (2000 - 1919), 0, 1).astype(np.float32),
        "title_svd": title_svd,
        "is_cold": is_cold,
    }


# ----------------------------------------------------------------------------- splitting
def split_ratings(ratings: pd.DataFrame, movies: pd.DataFrame, cfg,
                  flagged_user_ids: list[int]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Cold-item holdout + per-user time-aware split + index maps (spec 4.3 steps 3-5).

    The split is time-ordered within each user, not globally: train contains other users' ratings
    made after a given user's test period, so train-derived popularity/trending features can use
    information from after that period. A global-timestamp split would be stricter.

    Returns (ratings_split, movies_out, index_maps). Flagged users keep their rows and index;
    they are only excluded from model fitting downstream. `mean_train_rating` is NaN for movies
    without train ratings.
    """
    rng = np.random.default_rng(cfg.seed)
    counts = ratings["movie_id"].value_counts()
    candidates = np.sort(counts[counts >= cfg.cold_item_min_ratings].index.to_numpy())
    n_cold = int(round(cfg.cold_item_frac * len(candidates)))
    cold_items = np.sort(rng.choice(candidates, size=n_cold, replace=False)) if n_cold else np.array([], int)

    is_cold_row = ratings["movie_id"].isin(cold_items).to_numpy()
    warm = ratings.loc[~is_cold_row].sort_values(["user_id", "timestamp", "movie_id"], kind="stable")
    warm = warm.reset_index(drop=True)
    pos = warm.groupby("user_id").cumcount().to_numpy()
    n = warm.groupby("user_id")["movie_id"].transform("size").to_numpy()
    n_test = np.maximum(1, np.floor(cfg.test_frac * n)).astype(int)
    n_val = np.maximum(1, np.floor(cfg.val_frac * n)).astype(int)
    split = np.where(pos >= n - n_test, "test", np.where(pos >= n - n_test - n_val, "val", "train"))
    warm["split"] = split

    cold = ratings.loc[is_cold_row].sort_values(["user_id", "timestamp", "movie_id"], kind="stable").copy()
    cold["split"] = "cold"
    out = pd.concat([warm, cold], ignore_index=True)

    user_ids = np.sort(ratings["user_id"].unique())
    movie_ids = np.sort(movies["movie_id"].unique())
    missing = ~np.isin(out["movie_id"].to_numpy(), movie_ids)
    if missing.any():
        bad = np.unique(out["movie_id"].to_numpy()[missing])[:10].tolist()
        raise ValueError(f"ratings reference {int(missing.sum())} rows with movie_ids missing from movies, e.g. {bad}")
    out["u"] = np.searchsorted(user_ids, out["user_id"].to_numpy()).astype(np.int32)
    out["i"] = np.searchsorted(movie_ids, out["movie_id"].to_numpy()).astype(np.int32)
    out["rating"] = out["rating"].astype(np.float32)
    out = out[["user_id", "movie_id", "rating", "timestamp", "u", "i", "split"]]

    movies_out = movies.sort_values("movie_id").reset_index(drop=True).copy()
    if "year" not in movies_out.columns:
        movies_out = _add_year(movies_out)
    movies_out["i"] = np.arange(len(movies_out), dtype=np.int32)
    movies_out["is_cold"] = movies_out["movie_id"].isin(cold_items).to_numpy()
    train = out[out["split"] == "train"]
    agg = train.groupby("i")["rating"].agg(["size", "mean"])
    movies_out["n_train_ratings"] = agg["size"].reindex(movies_out["i"]).fillna(0).astype(int).to_numpy()
    movies_out["mean_train_rating"] = agg["mean"].reindex(movies_out["i"]).astype(float).to_numpy()
    movies_out = movies_out[["movie_id", "title", "year", "genres", "i", "is_cold",
                             "n_train_ratings", "mean_train_rating"]]

    index_maps = {
        "user_ids": [int(x) for x in user_ids],
        "movie_ids": [int(x) for x in movie_ids],
        "cold_movie_ids": [int(x) for x in cold_items],
        "flagged_user_ids": sorted(int(x) for x in flagged_user_ids),
    }
    return out, movies_out, index_maps


def split_stats(ratings_split: pd.DataFrame, index_maps: dict) -> dict:
    """Counts per split plus dataset sizes (written to split_stats.json)."""
    vc = ratings_split["split"].value_counts()
    stats = {f"n_{s}": int(vc.get(s, 0)) for s in SPLITS}
    stats.update({
        "n_ratings": int(len(ratings_split)),
        "n_users": len(index_maps["user_ids"]),
        "n_items": len(index_maps["movie_ids"]),
        "n_cold_items": len(index_maps["cold_movie_ids"]),
        "n_flagged": len(index_maps["flagged_user_ids"]),
    })
    return stats


# ----------------------------------------------------------------------------- artifact I/O
def _save_df(df: pd.DataFrame, path: Path) -> Path:
    """Save as parquet (pyarrow ships with streamlit); fall back to pickle next to it."""
    path = Path(path)
    alt = path.with_suffix(".pkl")
    try:
        df.to_parquet(path, index=False)
    except Exception:  # missing engine, ArrowNotImplementedError, ...: documented pickle fallback
        path.unlink(missing_ok=True)  # never leave a stale parquet that _load_df would prefer
        df.to_pickle(alt)
        return alt
    alt.unlink(missing_ok=True)  # drop a stale pickle from an earlier fallback run
    return path


def _load_df(path: Path) -> pd.DataFrame:
    """Load `<stem>.parquet`, or `<stem>.pkl` if only the pickle fallback exists."""
    path = Path(path)
    if path.exists():
        return pd.read_parquet(path)
    alt = path.with_suffix(".pkl")
    if alt.exists():
        return pd.read_pickle(alt)
    raise FileNotFoundError(path)


def _write_json(obj, path: Path) -> None:
    Path(path).write_text(json.dumps(obj, indent=2), encoding="utf-8")


def user_train_items(ratings_split: pd.DataFrame, n_users: int) -> list[np.ndarray]:
    """Per-user sorted TRAIN item indices (int32), one array per user index (empty if none)."""
    tr = ratings_split.loc[ratings_split["split"] == "train", ["u", "i"]]
    tr = tr.sort_values(["u", "i"], kind="stable")
    u = tr["u"].to_numpy()
    items = tr["i"].to_numpy(dtype=np.int32)
    indptr = np.searchsorted(u, np.arange(n_users + 1))
    return [items[indptr[k]:indptr[k + 1]] for k in range(n_users)]


def _to_csr_arrays(lists: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    indptr = np.zeros(len(lists) + 1, dtype=np.int64)
    indptr[1:] = np.cumsum([len(a) for a in lists])
    indices = np.concatenate(lists).astype(np.int32) if lists else np.zeros(0, np.int32)
    return indptr, indices


def prepare_data(cfg, flagged_user_ids, raw_dir: Path = RAW_DIR, processed_dir: Path = PROCESSED_DIR,
                 ratings: pd.DataFrame | None = None, movies: pd.DataFrame | None = None) -> dict:
    """Run spec steps 3-5 and write every section-4.5 artifact into `processed_dir`.

    `ratings`/`movies` may be passed pre-loaded (tests); otherwise they are read from `raw_dir`.
    Returns {"ratings", "movies", "item_features", "index_maps", "stats"}.
    """
    if ratings is None or movies is None:
        ratings, movies, _ = load_raw(raw_dir)
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    rs, movies_out, index_maps = split_ratings(ratings, movies, cfg, list(flagged_user_ids))
    item_features = build_item_features(movies_out, n_svd=cfg.tt_title_svd, seed=cfg.seed)
    stats = split_stats(rs, index_maps)
    indptr, indices = _to_csr_arrays(user_train_items(rs, stats["n_users"]))

    _save_df(rs, processed_dir / "ratings_split.parquet")
    _save_df(movies_out, processed_dir / "movies.parquet")
    np.savez(processed_dir / "item_features.npz", **item_features)
    np.savez(processed_dir / "user_train_items.npz", indptr=indptr, indices=indices)
    _write_json(index_maps, processed_dir / "index_maps.json")
    _write_json(stats, processed_dir / "split_stats.json")
    print(f"[data] wrote processed artifacts to {processed_dir}: {stats}")
    return {"ratings": rs, "movies": movies_out, "item_features": item_features,
            "index_maps": index_maps, "stats": stats}


def load_processed(processed_dir: Path = PROCESSED_DIR) -> dict:
    """Load the artifacts written by `prepare_data` (same keys as its return value)."""
    processed_dir = Path(processed_dir)
    try:
        rs = _load_df(processed_dir / "ratings_split.parquet")
        movies = _load_df(processed_dir / "movies.parquet")
        with np.load(processed_dir / "item_features.npz", allow_pickle=False) as z:
            item_features = {k: z[k] for k in z.files}
        index_maps = json.loads((processed_dir / "index_maps.json").read_text(encoding="utf-8"))
        stats = json.loads((processed_dir / "split_stats.json").read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Processed data missing in {processed_dir} ({exc}). "
            "Run: python scripts/download_data.py, then python scripts/train_all.py") from exc
    return {"ratings": rs, "movies": movies, "item_features": item_features,
            "index_maps": index_maps, "stats": stats}


# ----------------------------------------------------------------------------- views for models / eval
def train_matrix(ratings_split: pd.DataFrame, n_users: int, n_items: int, split: str = "train",
                 exclude_users=()) -> sp.csr_matrix:
    """CSR [n_users, n_items] of ratings in `split`, dropping rows of user INDICES in `exclude_users`."""
    df = ratings_split[ratings_split["split"] == split]
    if len(exclude_users):
        df = df[~df["u"].isin(np.asarray(list(exclude_users)))]
    return sp.csr_matrix((df["rating"].to_numpy(np.float32), (df["u"].to_numpy(), df["i"].to_numpy())),
                         shape=(n_users, n_items))


def holdout_dict(ratings_split: pd.DataFrame, split: str) -> dict[int, dict[int, float]]:
    """u -> {i: rating} for every row of `split`."""
    df = ratings_split[ratings_split["split"] == split]
    out: dict[int, dict[int, float]] = {}
    for u, i, r in zip(df["u"].to_numpy().tolist(), df["i"].to_numpy().tolist(),
                       df["rating"].to_numpy().tolist()):
        out.setdefault(u, {})[i] = float(r)
    return out


def eval_users_sample(ratings_split: pd.DataFrame, n: int | None, seed: int, exclude_users=()) -> np.ndarray:
    """Sorted user INDICES with >= 1 test row and not in `exclude_users`; seeded sample of n (None = all)."""
    users = np.unique(ratings_split.loc[ratings_split["split"] == "test", "u"].to_numpy())
    if len(exclude_users):
        users = users[~np.isin(users, np.asarray(list(exclude_users)))]
    if n is not None and n < len(users):
        users = np.random.default_rng(seed).choice(users, size=n, replace=False)
    return np.sort(users).astype(np.int64)


def flagged_indices(index_maps: dict) -> np.ndarray:
    """User INDICES of the flagged user_ids in `index_maps` (ids absent from the map are ignored)."""
    user_ids = np.asarray(index_maps["user_ids"])
    flagged = np.asarray(index_maps.get("flagged_user_ids", []), dtype=user_ids.dtype)
    return np.flatnonzero(np.isin(user_ids, flagged))


def _attach_eval_keys(data: dict, eval_users: np.ndarray) -> dict:
    """Add the ctx["data"] keys of spec 11.2 to a prepare_data/load_processed dict."""
    rs = data["ratings"]
    data["eval_users"] = eval_users
    data["train_items"] = user_train_items(rs, data["stats"]["n_users"])
    data["holdout_test"] = holdout_dict(rs, "test")
    data["holdout_cold"] = holdout_dict(rs, "cold")
    data["warm_mask"] = ~np.asarray(data["item_features"]["is_cold"], dtype=bool)
    return data


def _data_metrics(stats: dict, n_eval_users: int) -> dict:
    """The metrics.json "data" dict (spec 6.3)."""
    return {"n_users": stats["n_users"], "n_items": stats["n_items"], "n_cold_items": stats["n_cold_items"],
            "n_flagged_users": stats["n_flagged"], "n_train": stats["n_train"], "n_val": stats["n_val"],
            "n_test": stats["n_test"], "n_cold": stats["n_cold"], "n_eval_users": int(n_eval_users)}


def train_step(cfg, ctx: dict) -> dict:
    """Prepare processed data, write eval_users.npy (flagged users excluded) and set ctx["data"]."""
    raw_dir = Path(ctx.get("raw_dir", RAW_DIR))
    processed_dir = Path(ctx.get("processed_dir", PROCESSED_DIR))
    models_dir = Path(ctx.get("models_dir", MODELS_DIR))
    models_dir.mkdir(parents=True, exist_ok=True)
    data = prepare_data(cfg, ctx.get("flagged_user_ids", []), raw_dir=raw_dir, processed_dir=processed_dir)
    eval_users = eval_users_sample(data["ratings"], cfg.eval_users, cfg.seed,
                                   exclude_users=flagged_indices(data["index_maps"]))
    out_path = models_dir / "eval_users.npy"
    np.save(out_path, eval_users)
    ctx["data"] = _attach_eval_keys(data, eval_users)
    print(f"[data] {len(eval_users)} eval users -> {out_path}")
    return _data_metrics(data["stats"], len(eval_users))


def eval_step(cfg, ctx: dict) -> dict | None:
    """Load processed data + eval users into ctx["data"]; return the metrics.json "data" dict (None if missing)."""
    processed_dir = Path(ctx.get("processed_dir", PROCESSED_DIR))
    models_dir = Path(ctx.get("models_dir", MODELS_DIR))
    try:
        data = load_processed(processed_dir)
    except FileNotFoundError as exc:
        print(f"[data] {exc}")
        return None
    path = models_dir / "eval_users.npy"
    if path.exists():
        eval_users = np.load(path, allow_pickle=False)
    else:  # robustness when run alone: recompute the same seeded sample
        print(f"[data] {path} missing; recomputing the seeded eval-user sample")
        eval_users = eval_users_sample(data["ratings"], cfg.eval_users, cfg.seed,
                                       exclude_users=flagged_indices(data["index_maps"]))
    ctx["data"] = _attach_eval_keys(data, eval_users)
    return _data_metrics(data["stats"], len(eval_users))
