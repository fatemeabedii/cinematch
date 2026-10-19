"""Anomaly detection: flag bot-like users with a Gaussian density model.

Pipeline (spec sections 4.3 step 2, 8, 17.4 and 18.1):
  1. Compute the 5 behaviour features per user (FEATURES) from the RAW ratings (pre-split).
  2. Inject 5 types of synthetic bots (ids >= 100000) so we have labelled anomalies.
  3. Split real users 60/20/20 into normal_train / val_real / test_real; bots 50/50
     (stratified by type) into val_bots / test_bots.
  4. Standardise features with normal_train mean/std, fit an independent Gaussian per
     feature, score = sum of univariate log pdfs (log p(x)).
  5. Pick epsilon (flag if score < epsilon) by max F1 on val; report on test.
  6. Apply epsilon to ALL real users -> flagged_users.json, capped at 3% of real users.

Note: many real MovieLens-1M users rated everything in one session, so a high
ratings/day alone is only weakly anomalous; it is the combination of features
(e.g. high ratings/day AND extreme ratings) that makes a bot stand out.

Error analysis on the VALIDATION bots showed that two bot types, ``random_spam`` and
``camouflage_popular``, evade the 5-feature model. An engineered item-agreement feature,
``mean_abs_item_dev`` (how far a user's ratings sit from each movie's average), was tried:
the same protocol is re-run on FEATURES_PLUS = FEATURES + ["mean_abs_item_dev"] and stored as
the ablation ``ablation_plus_item_dev``. It did not help, because these bots mimic real users'
rating statistics; catching them would need item-level / co-rating graph signals (future
work). The DEPLOYED model (epsilon, 3% cap, flagged_users.json, anomaly.npz, figures) uses
the 5 FEATURES. Test bots come from the same synthetic generator, so the test numbers
measure detection of these patterns, not of real-world abuse.

Never imports torch (numpy/pandas only; matplotlib is imported lazily for figures).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from cinematch.config import FIGURES_DIR, MODELS_DIR, RAW_DIR, SEED

FEATURES = ["log_count", "mean_rating", "var_rating", "pct_extreme", "log_ratings_per_day"]
FEATURES_PLUS = FEATURES + ["mean_abs_item_dev"]   # ablation: + engineered item-agreement feature
BOT_TYPES = ["push", "nuke", "random_spam", "camouflage_popular", "camouflage_slow"]
BOT_ID_START = 100000
MAX_FLAG_FRAC = 0.03          # safety cap: never flag more than 3% of real users
N_TARGETS = 20                # shared "target" movies that push-style bots promote

_HOUR = 3600
_DAY = 86400
_BOT_WINDOWS = {               # all timestamps of one bot fall inside this window (seconds)
    "push": _HOUR,
    "nuke": _HOUR,
    "random_spam": _DAY,
    "camouflage_popular": 3 * _DAY,
    "camouflage_slow": 180 * _DAY,
}


def _log(msg: str) -> None:
    print(f"[anomaly] {msg}")


# ----------------------------------------------------------------------------- features
def user_features(ratings: pd.DataFrame, with_item_dev: bool = False) -> pd.DataFrame:
    """Per-user behaviour features (index user_id, columns FEATURES, or FEATURES_PLUS when
    `with_item_dev` is True).

    log_count = log1p(n); mean_rating; var_rating (population variance);
    pct_extreme = share of ratings in {1, 5};
    log_ratings_per_day = log1p(n / max(1, span_days + 1)), span_days = (max_ts - min_ts) / 86400;
    mean_abs_item_dev (ablation only) = mean over the user's ratings of |rating - item_mean|, where item_mean is
    the movie's mean rating over the SAME `ratings` frame (bots included when they are present,
    as on a live platform).
    """
    df = pd.DataFrame({
        "user_id": ratings["user_id"].to_numpy(),
        "rating": ratings["rating"].to_numpy(dtype=np.float64),
        "timestamp": ratings["timestamp"].to_numpy(dtype=np.float64),
    })
    df["extreme"] = df["rating"].isin([1.0, 5.0]).astype(np.float64)
    g = df.groupby("user_id", sort=True)
    n = g["rating"].size().astype(np.float64)
    span_days = (g["timestamp"].max() - g["timestamp"].min()) / _DAY
    out = pd.DataFrame({
        "log_count": np.log1p(n),
        "mean_rating": g["rating"].mean(),
        "var_rating": g["rating"].var(ddof=0),
        "pct_extreme": g["extreme"].mean(),
        "log_ratings_per_day": np.log1p(n / np.maximum(1.0, span_days + 1.0)),
    })
    out.index.name = "user_id"
    if not with_item_dev:
        return out[FEATURES]
    item_mean = df["rating"].groupby(ratings["movie_id"].to_numpy()).transform("mean")
    out["mean_abs_item_dev"] = (df["rating"] - item_mean).abs().groupby(df["user_id"]).mean()
    return out[FEATURES_PLUS]


# ----------------------------------------------------------------------------- bots
def inject_bots(ratings: pd.DataFrame, movies: pd.DataFrame, n_per_type: int,
                seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Append synthetic bot users to `ratings`.

    Returns (ratings_with_bots, bots_df[user_id, bot_type]). Bot ids are 100000 + k.
    Each bot rates n ~ randint(30, 300) distinct movies (capped by the pool size) with all
    timestamps inside a type-specific window (see spec 8.2 for the 5 types).
    """
    rng = np.random.default_rng(seed)
    all_movies = np.sort(movies["movie_id"].to_numpy())
    counts = ratings.groupby("movie_id")["rating"].agg(["size", "mean"])
    item_mean = counts["mean"]
    popular = counts.sort_values("size", ascending=False, kind="stable").index.to_numpy()[:500]
    targets = rng.choice(all_movies, size=min(N_TARGETS, len(all_movies)), replace=False)
    non_targets = np.setdiff1d(all_movies, targets)
    ts_min, ts_max = int(ratings["timestamp"].min()), int(ratings["timestamp"].max())

    rows, bot_rows = [], []
    k = 0
    for bot_type in BOT_TYPES:
        for _ in range(n_per_type):
            uid = BOT_ID_START + k
            k += 1
            n = int(rng.integers(30, 301))
            if bot_type in ("push", "nuke"):
                n_t = min(len(targets), n)
                others = rng.choice(non_targets, size=min(n - n_t, len(non_targets)), replace=False)
                mids = np.concatenate([targets[:n_t], others])
                good, bad = (5.0, 1.0) if bot_type == "push" else (1.0, 5.0)
                r = np.where(rng.random(len(mids)) < 0.95, good, bad)
                r[:n_t] = good                          # targets always get the "good" rating
            elif bot_type == "random_spam":
                mids = rng.choice(all_movies, size=min(n, len(all_movies)), replace=False)
                r = rng.integers(1, 6, size=len(mids)).astype(np.float64)
            elif bot_type == "camouflage_popular":
                n_t = min(len(targets), int(round(0.1 * n)))
                pool = np.setdiff1d(popular, targets[:n_t])
                pop = rng.choice(pool, size=min(n - n_t, len(pool)), replace=False)
                r_pop = np.clip(np.round(item_mean.loc[pop].to_numpy() + rng.normal(0, 0.3, len(pop))), 1, 5)
                mids = np.concatenate([pop, targets[:n_t]])
                r = np.concatenate([r_pop, np.full(n_t, 5.0)])
            else:  # camouflage_slow: all 5s, spread over 180 days
                mids = rng.choice(all_movies, size=min(n, len(all_movies)), replace=False)
                r = np.full(len(mids), 5.0)
            window = _BOT_WINDOWS[bot_type]
            start = int(rng.integers(ts_min, max(ts_min + 1, ts_max - window)))
            ts = start + np.sort(rng.integers(0, window, size=len(mids)))
            rows.append(pd.DataFrame({"user_id": uid, "movie_id": mids, "rating": r, "timestamp": ts}))
            bot_rows.append((uid, bot_type))

    bots = pd.concat(rows, ignore_index=True)
    bots = bots.astype({"user_id": ratings["user_id"].dtype, "movie_id": ratings["movie_id"].dtype,
                        "rating": ratings["rating"].dtype, "timestamp": ratings["timestamp"].dtype})
    base = ratings[["user_id", "movie_id", "rating", "timestamp"]]
    out = pd.concat([base, bots], ignore_index=True)
    bots_df = pd.DataFrame(bot_rows, columns=["user_id", "bot_type"])
    return out, bots_df


# ----------------------------------------------------------------------------- Gaussian model
def fit_gaussian(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-feature mean and (population) variance, variance floored at 1e-6."""
    X = np.asarray(X, dtype=np.float64)
    return X.mean(axis=0), np.maximum(X.var(axis=0), 1e-6)


def log_density(X: np.ndarray, mu: np.ndarray, var: np.ndarray) -> np.ndarray:
    """log p(x) = sum_j log N(x_j | mu_j, var_j) (independent features). Accepts [n, d] or [d]."""
    X = np.asarray(X, dtype=np.float64)
    per_feature = -0.5 * np.log(2.0 * np.pi * var) - (X - mu) ** 2 / (2.0 * var)
    return per_feature.sum(axis=-1)


def evaluate_threshold(scores: np.ndarray, y: np.ndarray, eps: float) -> dict:
    """Flag score < eps as anomalous (y=1). Returns precision, recall, f1, n_flagged."""
    pred = np.asarray(scores) < eps
    y = np.asarray(y).astype(bool)
    tp = int(np.sum(pred & y))
    fp = int(np.sum(pred & ~y))
    fn = int(np.sum(~pred & y))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "n_flagged": int(pred.sum())}


def select_epsilon(scores_val: np.ndarray, y_val: np.ndarray, n_grid: int = 200) -> tuple[float, float]:
    """Pick epsilon with max val F1 over a grid of `n_grid` quantiles of the val scores.

    Ties are broken towards the smaller epsilon (flags fewer users).
    """
    grid = np.unique(np.quantile(scores_val, np.linspace(0.0, 1.0, n_grid)))  # ascending
    best_eps, best_f1 = float(grid[0]), -1.0
    for eps in grid:
        f1 = evaluate_threshold(scores_val, y_val, eps)["f1"]
        if f1 > best_f1:          # strict ">" keeps the smallest epsilon on ties
            best_eps, best_f1 = float(eps), f1
    return best_eps, float(best_f1)


def cap_epsilon(eps: float, real_scores: np.ndarray, max_frac: float = MAX_FLAG_FRAC) -> tuple[float, bool]:
    """Safety cap: if eps flags more than `max_frac` of real users, move eps down to the
    `max_frac` quantile of real scores. The new eps is the order statistic s[floor(max_frac*n)],
    so `score < eps` flags at most floor(max_frac * n) real users (a linearly interpolated
    quantile could flag one user too many). Returns (eps, capped)."""
    real_scores = np.asarray(real_scores, dtype=np.float64)
    if np.mean(real_scores < eps) > max_frac:
        s = np.sort(real_scores)
        return float(s[int(np.floor(max_frac * len(s)))]), True
    return float(eps), False


# ----------------------------------------------------------------------------- helpers
def _r4(x: float) -> float:
    return round(float(x), 4)


def _split_users(real_ids: np.ndarray, bots_df: pd.DataFrame, cfg) -> dict:
    """Real users -> 60/20/20 (normal_train/val_real/test_real); bots 50/50 per type."""
    rng = np.random.default_rng(cfg.seed)
    perm = rng.permutation(real_ids)
    n_train = int(round(cfg.anomaly_train_frac * len(perm)))
    n_val = int(round(cfg.anomaly_val_frac * len(perm)))
    val_bots, test_bots = [], []
    for bot_type in BOT_TYPES:
        ids = rng.permutation(bots_df.loc[bots_df["bot_type"] == bot_type, "user_id"].to_numpy())
        half = len(ids) // 2
        val_bots.append(ids[:half])
        test_bots.append(ids[half:])
    return {
        "normal_train": np.sort(perm[:n_train]),
        "val_real": np.sort(perm[n_train:n_train + n_val]),
        "test_real": np.sort(perm[n_train + n_val:]),
        "val_bots": np.sort(np.concatenate(val_bots)),
        "test_bots": np.sort(np.concatenate(test_bots)),
    }


def _load_raw(ctx: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Raw ratings/movies. Test hook: ctx["raw_ratings"]/ctx["raw_movies"] skip the disk read."""
    if "raw_ratings" in ctx:
        return ctx["raw_ratings"], ctx["raw_movies"]
    from cinematch import data
    ratings, movies, _ = data.load_raw(Path(ctx.get("raw_dir", RAW_DIR)))
    return ratings, movies


# ----------------------------------------------------------------------------- pipeline steps
def train_step(cfg, ctx: dict) -> dict:
    """Fit the detector, choose epsilon, flag real users.

    The deployed model uses the 5 FEATURES; the same protocol is repeated with FEATURES_PLUS
    (+ mean_abs_item_dev) and stored as report["ablation_plus_item_dev"] (spec 18.1).
    Writes models/anomaly.npz, models/anomaly_report.json, models/flagged_users.json and sets
    ctx["flagged_user_ids"]. Returns the anomaly report dict.

    Caveat: features use each user's FULL history (the raw ratings, before the train/val/test
    split), so the flagged-user filter on the shared eval sample is not strictly blind to the
    held-out period.
    """
    models_dir = Path(ctx.get("models_dir", MODELS_DIR))
    models_dir.mkdir(parents=True, exist_ok=True)
    ratings, movies = _load_raw(ctx)

    with_bots, bots_df = inject_bots(ratings, movies, cfg.anomaly_bots_per_type, cfg.seed)
    feats = user_features(with_bots, with_item_dev=True)
    bot_type_of = dict(zip(bots_df["user_id"].to_numpy(), bots_df["bot_type"].to_numpy()))
    real_ids = feats.index.to_numpy()[feats.index.to_numpy() < BOT_ID_START]
    s = _split_users(real_ids, bots_df, cfg)
    _log(f"{len(real_ids)} real users, {len(bots_df)} bots; "
         f"train/val/test real = {len(s['normal_train'])}/{len(s['val_real'])}/{len(s['test_real'])}")

    det = _fit_detector(feats[FEATURES], s, real_ids, bot_type_of, cfg)
    abl = _fit_detector(feats[FEATURES_PLUS], s, real_ids, bot_type_of, cfg)
    eps, eps_val, capped, scores_real = det["eps"], det["eps_val"], det["capped"], det["scores_real"]
    if capped:
        _log(f"epsilon capped: {eps_val:.3f} -> {eps:.3f} (would flag > {MAX_FLAG_FRAC:.0%} of real users)")
    flagged = [int(u) for u in real_ids[scores_real < eps]] if cfg.anomaly_flag_real_users else []

    bot_scores = det["scores"].loc[bots_df["user_id"].to_numpy()].to_numpy()
    lo, hi = float(min(scores_real.min(), bot_scores.min())), float(max(scores_real.max(), bot_scores.max()))
    bins = np.linspace(lo, hi, 61)
    report = {
        "epsilon": _r4(eps),
        "epsilon_val": _r4(eps_val),
        "epsilon_capped": bool(capped),
        "val_f1": _r4(det["val_f1"]),
        "test_precision": _r4(det["test"]["precision"]),
        "test_recall": _r4(det["test"]["recall"]),
        "test_f1": _r4(det["test"]["f1"]),
        "n_bots_test": int(len(s["test_bots"])),
        "n_real_test": int(len(s["test_real"])),
        "n_real_flagged": det["n_real_flagged_test"],
        "n_real_flagged_all": len(flagged),
        "n_real_users": int(len(real_ids)),
        "per_bot_type_recall": {row["bot_type"]: row["recall"] for row in det["bot_table"]},
        "bot_type_table": det["bot_table"],
        "ablation_plus_item_dev": {
            "epsilon": _r4(abl["eps"]),
            "val_f1": _r4(abl["val_f1"]),
            "test_precision": _r4(abl["test"]["precision"]),
            "test_recall": _r4(abl["test"]["recall"]),
            "test_f1": _r4(abl["test"]["f1"]),
            "per_bot_type_recall": {row["bot_type"]: row["recall"] for row in abl["bot_table"]},
        },
        "ablation_plus_item_dev_capped": bool(abl["capped"]),
        "val_per_bot_type_recall": det["val_per_bot_type_recall"],
        "ablation_plus_item_dev_val_per_bot_type_recall": abl["val_per_bot_type_recall"],
        "features": FEATURES,
        "ablation_features": FEATURES_PLUS,
        "split_sizes": {k: int(len(v)) for k, v in s.items()},
        "histogram": {"bins": [float(b) for b in bins],
                      "real_counts": np.histogram(scores_real, bins)[0].tolist(),
                      "bot_counts": np.histogram(bot_scores, bins)[0].tolist()},
    }

    np.savez(models_dir / "anomaly.npz", mu=det["mu"], var=det["var"], eps=np.float64(eps),
             eps_val=np.float64(eps_val), feat_mean=det["feat_mean"], feat_std=det["feat_std"],
             real_user_ids=real_ids.astype(np.int64), scores_real=scores_real,
             scores_test_bots=det["scores_test_bots"], test_bot_types=det["test_bot_types"],
             scores_val=det["scores_val"], y_val=det["y_val"], scores_test=det["scores_test"],
             y_test=det["y_test"], scores_all_bots=bot_scores,
             features=np.array(FEATURES, dtype="U32"))
    (models_dir / "anomaly_report.json").write_text(json.dumps(report, indent=2))
    (models_dir / "flagged_users.json").write_text(json.dumps(flagged))
    ctx["flagged_user_ids"] = flagged
    _log(f"deployed ({len(FEATURES)} features): eps={eps:.3f} (capped={capped}) val F1={det['val_f1']:.3f} "
         f"test F1={det['test']['f1']:.3f}; flagged {len(flagged)} real users")
    _log(f"ablation (+ mean_abs_item_dev): eps={abl['eps']:.3f} (capped={abl['capped']}) val F1={abl['val_f1']:.3f} "
         f"test F1={abl['test']['f1']:.3f}")
    return report


def _fit_detector(feats: pd.DataFrame, s: dict, real_ids: np.ndarray, bot_type_of: dict, cfg) -> dict:
    """Run the protocol on one feature set (the columns of `feats`).

    Standardise with normal_train statistics, fit the Gaussian on the same users, pick epsilon
    by val F1, apply the 3% cap over ALL real users, and evaluate on test at the final epsilon.
    Shared by the deployed 5-feature model and the + item-dev ablation; returns unrounded values.
    """
    X_train = feats.loc[s["normal_train"]].to_numpy(np.float64)
    feat_mean = X_train.mean(axis=0)
    feat_std = X_train.std(axis=0)
    feat_std = np.where(feat_std < 1e-6, 1.0, feat_std)
    Z = (feats.to_numpy(np.float64) - feat_mean) / feat_std
    mu, var = fit_gaussian(Z[np.isin(feats.index.to_numpy(), s["normal_train"])])
    scores = pd.Series(log_density(Z, mu, var), index=feats.index)

    # Epsilon on validation (real = 0, bot = 1).
    val_ids = np.concatenate([s["val_real"], s["val_bots"]])
    y_val = np.concatenate([np.zeros(len(s["val_real"])), np.ones(len(s["val_bots"]))]).astype(np.int8)
    scores_val = scores.loc[val_ids].to_numpy()
    eps_val, val_f1 = select_epsilon(scores_val, y_val, cfg.anomaly_eps_grid)

    # 3% safety cap computed over ALL real users (the exclusion hook uses this epsilon).
    scores_real = scores.loc[real_ids].to_numpy()
    eps, capped = cap_epsilon(eps_val, scores_real)

    # Test-set report at the final (deployed) epsilon. Caveat: the cap above uses all real users,
    # test_real included, so IF it fires the test metrics depend on a threshold partly set from
    # test-user scores. When the cap does not fire (epsilon_capped=false), eps == eps_val is chosen
    # on validation only and the test metrics are clean.
    test_ids = np.concatenate([s["test_real"], s["test_bots"]])
    y_test = np.concatenate([np.zeros(len(s["test_real"])), np.ones(len(s["test_bots"]))]).astype(np.int8)
    scores_test = scores.loc[test_ids].to_numpy()
    test = evaluate_threshold(scores_test, y_test, eps)
    scores_test_bots = scores.loc[s["test_bots"]].to_numpy()
    test_bot_types = np.array([bot_type_of[u] for u in s["test_bots"]], dtype="U20")
    bot_table = _bot_type_table(scores_test_bots, test_bot_types, eps, "n_test")
    # Same per-type recall on the VALIDATION bots (the error analysis that motivated the ablation).
    val_bot_types = np.array([bot_type_of[u] for u in s["val_bots"]], dtype="U20")
    val_bot_table = _bot_type_table(scores.loc[s["val_bots"]].to_numpy(), val_bot_types, eps, "n_val")
    return {
        "feat_mean": feat_mean, "feat_std": feat_std, "mu": mu, "var": var, "scores": scores,
        "eps_val": eps_val, "val_f1": val_f1, "eps": eps, "capped": capped,
        "scores_val": scores_val, "y_val": y_val, "scores_real": scores_real,
        "scores_test": scores_test, "y_test": y_test, "test": test,
        "scores_test_bots": scores_test_bots, "test_bot_types": test_bot_types, "bot_table": bot_table,
        "val_per_bot_type_recall": {row["bot_type"]: row["recall"] for row in val_bot_table},
        "n_real_flagged_test": int(np.sum(scores.loc[s["test_real"]].to_numpy() < eps)),
    }


def _bot_type_table(bot_scores: np.ndarray, bot_types: np.ndarray, eps: float, n_key: str) -> list[dict]:
    """Per-bot-type detection counts and recall (flag if score < eps)."""
    table = []
    for bot_type in BOT_TYPES:
        m = bot_types == bot_type
        n_det = int(np.sum(bot_scores[m] < eps))
        table.append({"bot_type": bot_type, n_key: int(m.sum()), "n_detected": n_det,
                      "recall": _r4(n_det / m.sum()) if m.sum() else 0.0})
    return table


def _plot_figures(art: dict, report: dict, figures_dir: Path, n_grid: int) -> None:
    """anomaly_scores.png (real vs bot histogram + epsilon) and anomaly_pr_curve.png (val P/R/F1 vs eps)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir.mkdir(parents=True, exist_ok=True)
    eps = float(art["eps"])
    bins = np.asarray(report["histogram"]["bins"])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(art["scores_real"], bins=bins, density=True, alpha=0.6, label="real users")
    ax.hist(art["scores_all_bots"], bins=bins, density=True, alpha=0.6, label="synthetic bots")
    ax.axvline(eps, color="black", linestyle="--", label=f"epsilon = {eps:.2f}")
    ax.set_xlabel("log density  log p(x)")
    ax.set_ylabel("density")
    ax.set_title("Anomaly scores: real users vs synthetic bots (flag if score < epsilon)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "anomaly_scores.png", dpi=150)
    plt.close(fig)

    grid = np.unique(np.quantile(art["scores_val"], np.linspace(0.0, 1.0, n_grid)))
    curves = [evaluate_threshold(art["scores_val"], art["y_val"], e) for e in grid]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for key in ("precision", "recall", "f1"):
        ax.plot(grid, [c[key] for c in curves], label=key)
    ax.axvline(float(art["eps_val"]), color="gray", linestyle=":", label="best val F1")
    if report.get("epsilon_capped"):
        ax.axvline(eps, color="black", linestyle="--", label="capped epsilon")
    ax.set_xlabel("epsilon")
    ax.set_ylabel("score on validation set")
    ax.set_title("Choosing epsilon on validation (real users + bots)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "anomaly_pr_curve.png", dpi=150)
    plt.close(fig)


def eval_step(cfg, ctx: dict) -> dict | None:
    """Read anomaly.npz + anomaly_report.json, draw figures, return the metrics.json "anomaly" dict.

    Returns None if the artifacts are missing (partial pipeline).
    """
    models_dir = Path(ctx.get("models_dir", MODELS_DIR))
    figures_dir = Path(ctx.get("figures_dir", FIGURES_DIR))
    npz_path, report_path = models_dir / "anomaly.npz", models_dir / "anomaly_report.json"
    if not (npz_path.exists() and report_path.exists()):
        _log("artifacts missing; skipping eval")
        return None
    with np.load(npz_path, allow_pickle=False) as f:
        art = {k: f[k] for k in f.files}
    report = json.loads(report_path.read_text())
    _plot_figures(art, report, figures_dir, cfg.anomaly_eps_grid)
    keys = ["epsilon", "epsilon_capped", "val_f1", "test_precision", "test_recall", "test_f1",
            "n_bots_test", "n_real_flagged", "n_real_flagged_all", "per_bot_type_recall",
            "ablation_plus_item_dev", "ablation_plus_item_dev_capped"]
    return {k: report.get(k) for k in keys}


# ----------------------------------------------------------------------------- inference (app)
def load_anomaly(models_dir: Path = MODELS_DIR) -> dict:
    """Numpy-only loader: {"mu","var","eps","feat_mean","feat_std","features"}.

    "features" are the names the model was trained on (stored in the npz); a file without them
    raises a "retrain" ValueError. Callers must build the feature vector in model["features"] order.
    """
    path = Path(models_dir) / "anomaly.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python scripts/train_all.py")
    with np.load(path, allow_pickle=False) as f:
        model = {k: f[k] for k in ("mu", "var", "feat_mean", "feat_std")}
        model["eps"] = float(f["eps"])
        if "features" not in f.files:
            raise ValueError(f"{path} has no stored feature names (outdated artifact); "
                             "retrain with: python scripts/train_all.py")
        model["features"] = [str(x) for x in f["features"]]
    return model


def score_user(model: dict, feats: np.ndarray) -> float:
    """Log density of one user's RAW feature vector, in model["features"] order (FEATURES for the
    deployed model); anomalous if < model["eps"]."""
    x = np.asarray(feats, dtype=np.float64)
    n_model = len(np.atleast_1d(model["feat_mean"]))
    if x.shape[-1] != n_model:
        raise ValueError(f"score_user got {x.shape[-1]} features but the anomaly model was trained with "
                         f"{n_model} ({model.get('features', FEATURES)}); retrain with: python scripts/train_all.py")
    z = (x - model["feat_mean"]) / model["feat_std"]
    return float(log_density(z, model["mu"], model["var"]))
