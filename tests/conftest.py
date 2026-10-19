"""Shared pytest setup: sys.path shim + tiny synthetic MovieLens-like fixtures (built in milliseconds)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from cinematch.config import GENRES, Config  # noqa: E402

N_USERS, N_MOVIES, PER_USER, RANK = 200, 150, 30, 4
T0 = 978_300_000  # ML-1M era (late 2000)


def make_tiny_movies(seed: int = 0) -> pd.DataFrame:
    """Raw-format movies (movie_id, title, genres); ids are non-contiguous to exercise the index maps."""
    rng = np.random.default_rng(seed)
    ids = 10 + 3 * np.arange(N_MOVIES)
    genres = ["|".join(rng.choice(GENRES, size=rng.integers(1, 4), replace=False)) for _ in ids]
    titles = [f"Movie {m} (199{j % 10})" for j, m in enumerate(ids)]
    return pd.DataFrame({"movie_id": ids, "title": titles, "genres": genres})


def make_tiny_ratings(seed: int = 0) -> pd.DataFrame:
    """~6,000 ratings (200 users x 30 movies) from a planted rank-4 model + noise, timestamps over 400 days."""
    rng = np.random.default_rng(seed)
    movie_ids = 10 + 3 * np.arange(N_MOVIES)
    P = rng.normal(0, 0.6, (N_USERS, RANK))
    Q = rng.normal(0, 0.6, (N_MOVIES, RANK))
    item_bias = rng.normal(0, 0.4, N_MOVIES)
    pop = np.linspace(2.0, 0.5, N_MOVIES)
    pop /= pop.sum()
    rows = []
    for u in range(N_USERS):
        items = rng.choice(N_MOVIES, size=PER_USER, replace=False, p=pop)
        r = 3.6 + item_bias[items] + Q[items] @ P[u] + rng.normal(0, 0.3, PER_USER)
        ts = T0 + rng.integers(0, 400 * 86400, size=PER_USER)
        for j, it in enumerate(items):
            rows.append((u + 1, int(movie_ids[it]), float(np.clip(np.rint(r[j]), 1, 5)), int(ts[j])))
    df = pd.DataFrame(rows, columns=["user_id", "movie_id", "rating", "timestamp"])
    df["rating"] = df["rating"].astype(np.float32)
    return df


def write_raw_dat(ratings: pd.DataFrame, movies: pd.DataFrame, raw_dir: Path) -> Path:
    """Write ratings/movies/users .dat files in the ML-1M '::' format (latin-1)."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    with open(raw_dir / "ratings.dat", "w", encoding="latin-1") as f:
        for row in ratings.itertuples(index=False):
            f.write(f"{row.user_id}::{row.movie_id}::{int(row.rating)}::{row.timestamp}\n")
    with open(raw_dir / "movies.dat", "w", encoding="latin-1") as f:
        for row in movies.itertuples(index=False):
            f.write(f"{row.movie_id}::{row.title}::{row.genres}\n")
    with open(raw_dir / "users.dat", "w", encoding="latin-1") as f:
        for uid in sorted(ratings["user_id"].unique()):
            f.write(f"{uid}::M::25::1::00000\n")
    return raw_dir


@pytest.fixture
def tiny_ratings() -> pd.DataFrame:
    return make_tiny_ratings()


@pytest.fixture
def tiny_movies() -> pd.DataFrame:
    return make_tiny_movies()


@pytest.fixture
def tiny_cfg() -> Config:
    """FAST config adapted to the tiny data (set after __post_init__, which would override eval_users)."""
    cfg = Config(fast=True)
    cfg.cold_item_min_ratings = 5
    cfg.anomaly_bots_per_type = 10
    cfg.eval_users = 50
    return cfg


@pytest.fixture
def tmp_models_dir(tmp_path) -> Path:
    d = tmp_path / "models"
    d.mkdir()
    return d


@pytest.fixture
def tmp_processed_dir(tmp_path) -> Path:
    d = tmp_path / "processed"
    d.mkdir()
    return d


@pytest.fixture
def tmp_figures_dir(tmp_path) -> Path:
    d = tmp_path / "figures"
    d.mkdir()
    return d


@pytest.fixture
def tiny_raw_dir(tmp_path, tiny_ratings, tiny_movies) -> Path:
    """tmp `ml-1m/` dir holding the tiny data as .dat files (for train_step via ctx["raw_dir"])."""
    return write_raw_dat(tiny_ratings, tiny_movies, tmp_path / "ml-1m")


@pytest.fixture
def tiny_ctx(tiny_raw_dir, tmp_processed_dir, tmp_models_dir, tmp_figures_dir) -> dict:
    """ctx with every directory override pointing into tmp_path (spec 11.2)."""
    return {"flagged_user_ids": [], "timings": {}, "raw_dir": tiny_raw_dir,
            "processed_dir": tmp_processed_dir, "models_dir": tmp_models_dir, "figures_dir": tmp_figures_dir}
