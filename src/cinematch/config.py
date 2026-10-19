"""Central configuration: paths, constants and the `Config` dataclass.

This module must stay importable without loading torch (see `set_seed`).
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]  # repo root
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "ml-1m"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
README_PATH = ROOT / "README.md"
ML1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
ML1M_ZIP = DATA_DIR / "ml-1m.zip"
ML1M_EXPECTED_FILES = ("ratings.dat", "movies.dat", "users.dat")

SEED = 42
TOP_K = 10
TORCH_THREADS = 8  # fixed for reproducibility, see set_seed
RELEVANT_THRESHOLD = 4  # rating >= 4 is relevant
GENRES = ["Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary", "Drama",
          "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War",
          "Western"]  # 18, ML-1M order


@dataclass
class Config:
    """All hyperparameters in one place. `fast=True` (or CINEMATCH_FAST=1) shrinks everything for smoke runs."""

    seed: int = SEED
    fast: bool = False                      # set by --fast or CINEMATCH_FAST=1
    # split
    test_frac: float = 0.10
    val_frac: float = 0.10
    cold_item_frac: float = 0.06
    cold_item_min_ratings: int = 20
    # anomaly
    anomaly_bots_per_type: int = 60         # 5 types -> 300 bots
    anomaly_train_frac: float = 0.60        # of real users (normal)
    anomaly_val_frac: float = 0.20          # rest 0.20 is test
    anomaly_eps_grid: int = 200             # candidate epsilons (quantiles of val scores)
    anomaly_flag_real_users: bool = True
    # baseline
    bias_reg: float = 10.0
    bias_iters: int = 10
    # CF MF
    cf_dim: int = 32
    cf_reg: float = 0.05                    # L2 weight on vectors+biases (per-sample, in loss); picked by val RMSE
    cf_lr: float = 2e-3
    cf_batch: int = 4096
    cf_epochs: int = 30
    cf_patience: int = 3
    # CF ranking (all chosen on the VALIDATION split by NDCG@10 inside cf.train_step, never on test)
    cf_rank_pop_betas: tuple = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)  # MF rank score + beta*log1p(train count)
    knn_neighbors: tuple = (50, 100, 200)    # item-kNN: neighbours kept per item
    knn_pop_gammas: tuple = (0.0, 0.25, 0.5)  # item-kNN rank score + gamma*log1p(train count)
    # new users (app "rate a few movies" flow) and its offline simulation
    new_user_min_train_ratings: int = 100   # warm items need this many train ratings to be shown to new users
    new_user_seed_ratings: int = 5          # ratings a simulated new user gives
    new_user_eval_users: int = 1000         # eval users replayed as new users
    # two-tower
    tt_hidden: int = 64
    tt_dim: int = 32
    tt_title_svd: int = 16
    tt_lr: float = 2e-3
    tt_batch: int = 4096
    tt_epochs: int = 20
    tt_patience: int = 3
    tt_weight_decay: float = 1e-5
    # clustering
    k_min: int = 4                          # >= 4 profiles for a useful cold-start choice (silhouette alone picks 2)
    k_max: int = 12
    k_cap: int = 8
    k_min_cluster_frac: float = 0.03
    cluster_name_top_n_movies: int = 300
    cluster_taste_weight: float = 2.0       # profile score = mu + bi + w_taste*(Q @ centroid) + w_pop*log1p(count)
    cluster_pop_weight: float = 0.5
    # bandit
    # rewards are only a few percent: with 1,000 users, 3,000 rounds is ~3 visits per user, too short
    # for any policy to separate the arms
    bandit_rounds: int = 20000
    bandit_seeds: tuple = (0, 1, 2, 3, 4)
    bandit_users: int = 1000                # eval users sampled for the simulator
    bandit_epsilon: float = 0.1
    bandit_alpha: float = 1.0               # LinUCB
    bandit_ridge: float = 1.0
    bandit_alpha_grid: tuple = (1.0, 0.3, 0.1, 0.03, 0.01)    # LinUCB alpha, tuned on disjoint users
    bandit_epsilon_grid: tuple = (0.2, 0.1, 0.05, 0.02)       # epsilon-greedy, tuned on disjoint users
    bandit_tune_users: int = 1000           # tuning users: not flagged, not in eval_users.npy
    bandit_tune_seeds: tuple = (100, 101)   # disjoint from bandit_seeds
    click_probs: dict = field(default_factory=lambda: {1: 0.02, 2: 0.05, 3: 0.15, 4: 0.45, 5: 0.75})
    click_unrated: float = 0.01
    # dqn (stretch)
    dqn_episodes: int = 3000                # 5-step sessions; keeps the step well under 90 s
    dqn_position_decay: float = 0.85        # click prob x decay**t at slot t
    dqn_fatigue: float = 0.8                # click prob x fatigue**(shown items sharing a genre)
    dqn_hidden: int = 128
    dqn_gamma: float = 0.9
    dqn_lr: float = 1e-3
    dqn_batch: int = 64
    dqn_buffer: int = 20000                 # replay capacity (transitions)
    dqn_warmup: int = 500                   # transitions collected before the first update
    dqn_target_sync: int = 200              # gradient steps between target-network syncs
    dqn_eps_end: float = 0.05
    dqn_eps_decay_frac: float = 0.5         # epsilon 1.0 -> eps_end over this fraction of episodes
    dqn_eval_every: int = 100               # episodes between greedy-policy checkpoints (curve only)
    # eval
    eval_users: int | None = 2000           # subsample of users for ranking metrics (None = all)
    eval_rank_batch: int = 256

    def __post_init__(self):
        if os.environ.get("CINEMATCH_FAST") == "1":
            self.fast = True
        if self.fast:
            self.cf_epochs = 2
            self.tt_epochs = 2
            self.k_max = 5
            self.bandit_rounds = 300
            self.bandit_seeds = (0, 1)
            self.bandit_users = 200
            self.bandit_tune_users = 100
            self.bandit_tune_seeds = (100,)
            self.eval_users = 300
            self.anomaly_bots_per_type = 20
            self.dqn_episodes = 200
            self.dqn_eval_every = 50
            self.new_user_eval_users = 100
            self.cf_rank_pop_betas = (0.0, 1.0)
            self.knn_neighbors = (50,)
            self.knn_pop_gammas = (0.0, 0.5)


def get_config(fast: bool | None = None) -> Config:
    """Return a Config; `fast=None` leaves the decision to the CINEMATCH_FAST env var."""
    return Config(fast=bool(fast)) if fast is not None else Config()


def ensure_dirs() -> None:
    """Create the processed-data, models and figures directories if missing."""
    for d in (PROCESSED_DIR, MODELS_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = SEED) -> None:
    """Seed python, numpy and torch, and pin torch to TORCH_THREADS threads.

    Torch is imported lazily so this module never loads it at import. The thread count is pinned
    because float reductions depend on it: results are bit-reproducible only with the same torch
    build and TORCH_THREADS (a different OS or torch build regenerates slightly different two-tower and
    bandit numbers).
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a hard dependency, but keep config usable without it
        return
    torch.manual_seed(seed)
    torch.set_num_threads(TORCH_THREADS)
