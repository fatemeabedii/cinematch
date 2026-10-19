"""Two-tower content model: item metadata tower x user rating-profile tower, trained on rating MSE.

Item features: genre multi-hot (18) + scaled year (1) + title TF-IDF->SVD (16) = 35 dims.
User features (40 dims, fixed layout): [0:18] genre affinity = mean of (r - 3) * genre,
[18:36] genre share = fraction of the user's rated movies having each genre, 36 mean rating,
37 std rating (population), 38 log1p(count), 39 mean scaled year of rated movies.

Because items are described only by metadata, the model can score cold items it never saw in
training. Torch is imported lazily (spec section 3); inference helpers are numpy-only.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import FIGURES_DIR, MODELS_DIR, PROCESSED_DIR, TOP_K

USER_FEAT_DIM = 18 + 18 + 4
ITEM_FEAT_DIM = 18 + 1 + 16


def _torch():
    import torch
    return torch


# ----------------------------------------------------------------------------- features (numpy)
def item_matrix(item_feats: dict) -> np.ndarray:
    """Item feature matrix [I, 35] = concat(genre_multi_hot, year_scaled[:, None], title_svd)."""
    return np.concatenate([
        np.asarray(item_feats["genre_multi_hot"], dtype=np.float32),
        np.asarray(item_feats["year_scaled"], dtype=np.float32)[:, None],
        np.asarray(item_feats["title_svd"], dtype=np.float32),
    ], axis=1).astype(np.float32)


# Profiles are built from 39 per-user sums: [0:18] sum (r-3)*g, [18:36] sum g, 36 sum r, 37 sum r^2,
# 38 sum year. Sums are accumulated column by column with np.bincount (numpy only, O(n_rows) memory).
_N_SUMS = 18 + 18 + 3


def _user_sums(u: np.ndarray, i: np.ndarray, r: np.ndarray, n_users: int,
               item_feats: dict) -> tuple[np.ndarray, np.ndarray]:
    """Per-user sums [n_users, 39] (layout above) and row counts [n_users] from rating rows (u, i, r)."""
    u = np.asarray(u, dtype=np.int64)
    i = np.asarray(i, dtype=np.int64)
    r = np.asarray(r, dtype=np.float64)
    G = np.asarray(item_feats["genre_multi_hot"], dtype=np.float64)
    year = np.asarray(item_feats["year_scaled"], dtype=np.float64)
    count = np.bincount(u, minlength=n_users).astype(np.float64)
    sums = np.zeros((n_users, _N_SUMS), dtype=np.float64)
    rc = r - 3.0
    for k in range(18):
        gk = G[i, k]
        sums[:, k] = np.bincount(u, weights=rc * gk, minlength=n_users)
        sums[:, 18 + k] = np.bincount(u, weights=gk, minlength=n_users)
    sums[:, 36] = np.bincount(u, weights=r, minlength=n_users)
    sums[:, 37] = np.bincount(u, weights=r * r, minlength=n_users)
    sums[:, 38] = np.bincount(u, weights=year[i], minlength=n_users)
    return sums, count


def _row_contrib(i: np.ndarray, r: np.ndarray, item_feats: dict) -> np.ndarray:
    """Each row's own contribution [n_rows, 39] to the per-user sums (used for leave-one-out profiles)."""
    g = np.asarray(item_feats["genre_multi_hot"], dtype=np.float64)[i]
    year = np.asarray(item_feats["year_scaled"], dtype=np.float64)[i]
    r = np.asarray(r, dtype=np.float64)
    return np.column_stack([(r - 3.0)[:, None] * g, g, r, r * r, year])


def _finalize(sums: np.ndarray, count: np.ndarray) -> np.ndarray:
    """Turn per-user sums [n, 39] and counts [n] into raw [n, 40] profiles (rows with count 0 -> zeros)."""
    has = count > 0
    means = np.where(has[:, None], sums / np.maximum(count, 1)[:, None], 0.0)
    out = np.zeros((len(count), USER_FEAT_DIM), dtype=np.float64)
    out[:, 0:36] = means[:, 0:36]
    out[:, 36] = means[:, 36]
    out[:, 37] = np.sqrt(np.maximum(means[:, 37] - means[:, 36] ** 2, 0.0))
    out[:, 38] = np.log1p(np.maximum(count, 0))
    out[:, 39] = means[:, 38]
    return out.astype(np.float32)


def _profiles(u: np.ndarray, i: np.ndarray, r: np.ndarray, n_users: int, item_feats: dict) -> np.ndarray:
    """Aggregate rating rows (user index u, item index i, rating r) into raw [n_users, 40] profiles."""
    return _finalize(*_user_sums(u, i, r, n_users, item_feats))   # users without rows stay all-zero


def user_profile_features(ratings_split: pd.DataFrame, item_feats: dict, n_users: int,
                          split: str = "train") -> np.ndarray:
    """Raw user profile features [U, 40] from the rows of `split`; zeros for users with no rows."""
    df = ratings_split[ratings_split["split"] == split]
    return _profiles(df["u"].to_numpy(), df["i"].to_numpy(), df["rating"].to_numpy(), n_users, item_feats)


def profile_from_ratings(ratings: dict[int, float], item_feats: dict) -> np.ndarray:
    """Raw [40] profile for a new user; `ratings` maps ITEM INDEX i -> rating (same code path as above)."""
    items = np.fromiter(ratings.keys(), dtype=np.int64, count=len(ratings))
    values = np.fromiter(ratings.values(), dtype=np.float64, count=len(ratings))
    return _profiles(np.zeros(len(items), dtype=np.int64), items, values, 1, item_feats)[0]


# ----------------------------------------------------------------------------- torch model
def build_two_tower(hidden: int = 64, dim: int = 32, mu: float = 0.0):
    """Two MLP towers (in -> hidden -> ReLU -> dim); prediction = mu + <user_vec, item_vec>."""
    torch = _torch()
    nn = torch.nn

    class TwoTower(nn.Module):
        def __init__(self):
            super().__init__()
            self.user_tower = nn.Sequential(nn.Linear(USER_FEAT_DIM, hidden), nn.ReLU(), nn.Linear(hidden, dim))
            self.item_tower = nn.Sequential(nn.Linear(ITEM_FEAT_DIM, hidden), nn.ReLU(), nn.Linear(hidden, dim))
            self.register_buffer("mu", torch.tensor(float(mu), dtype=torch.float32))

        def forward(self, user_x, item_x):
            return self.mu + (self.user_tower(user_x) * self.item_tower(item_x)).sum(dim=1)

    return TwoTower()


def _standardizer(user_feat_raw: np.ndarray, fit_users: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean/std over `fit_users` rows (std floored at 1e-6 -> 1.0 for constant columns)."""
    X = user_feat_raw[fit_users].astype(np.float64)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def train_two_tower(data: dict, cfg, exclude_users, log=print):
    """Train on TRAIN rows of non-excluded users; early stop on val RMSE (all users), keep best weights.

    Returns (model, history) with history = {"train_rmse": [...], "val_rmse": [...], "best_epoch": int
    (1-based)} (JSON-safe). The inputs export_two_tower needs are attached to the model as
    `model.export_inputs = {"feat_mean", "feat_std", "item_X", "user_feat_raw"}` (numpy arrays).

    Training batches use leave-one-out user profiles: each train row's own rating is removed from its
    user's profile, so the features never contain the label being predicted (as at val/test/serving
    time). Validation and the exported `user_feat_raw` use the full train profile.
    """
    torch = _torch()
    rs = data["ratings"]
    item_feats = data["item_features"]
    n_users = len(data["index_maps"]["user_ids"])
    exclude = np.asarray(list(exclude_users), dtype=np.int64)

    item_X = item_matrix(item_feats)
    all_train = rs[rs["split"] == "train"]
    sums, count = _user_sums(all_train["u"].to_numpy(), all_train["i"].to_numpy(),
                             all_train["rating"].to_numpy(), n_users, item_feats)
    user_raw = _finalize(sums, count)                       # == user_profile_features(rs, ..., "train")
    train = all_train[~all_train["u"].isin(exclude)]
    val = rs[rs["split"] == "val"]
    fit_users = np.setdiff1d(np.unique(train["u"].to_numpy()), exclude)
    feat_mean, feat_std = _standardizer(user_raw, fit_users)

    def t(a, dtype):  # copy: pandas (copy-on-write) hands out read-only arrays
        return torch.from_numpy(np.array(a, dtype=dtype, copy=True))

    U = t((user_raw - feat_mean) / feat_std, np.float32)
    IX = t(item_X, np.float32)
    tu_np = train["u"].to_numpy().astype(np.int64)
    ti_np = train["i"].to_numpy().astype(np.int64)
    tr_np = train["rating"].to_numpy().astype(np.float64)
    ti, tr = t(ti_np, np.int64), t(tr_np, np.float32)
    vu, vi, vr = t(val["u"], np.int64), t(val["i"], np.int64), t(val["rating"], np.float32)

    def loo_user_x(rows: np.ndarray):
        """Standardized leave-one-out profiles [len(rows), 40] for the given train rows."""
        bu, bi, br = tu_np[rows], ti_np[rows], tr_np[rows]
        raw = _finalize(sums[bu] - _row_contrib(bi, br, item_feats), count[bu] - 1.0)
        return torch.from_numpy((raw - feat_mean) / feat_std)

    mu = float(tr.mean()) if len(tr) else 3.5
    torch.manual_seed(cfg.seed)
    model = build_two_tower(cfg.tt_hidden, cfg.tt_dim, mu)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.tt_lr, weight_decay=cfg.tt_weight_decay)
    gen = torch.Generator().manual_seed(cfg.seed)

    def val_rmse() -> float:
        model.eval()
        with torch.no_grad():
            preds = [model(U[vu[s:s + 65536]], IX[vi[s:s + 65536]]) for s in range(0, len(vu), 65536)]
            pred = torch.cat(preds).clamp(1.0, 5.0) if preds else torch.zeros(0)
        model.train()
        return float(torch.sqrt(((pred - vr) ** 2).mean()))

    history = {"train_rmse": [], "val_rmse": [], "best_epoch": 0}
    best_val, best_state, bad = float("inf"), None, 0
    for epoch in range(1, cfg.tt_epochs + 1):
        t0 = time.time()
        perm = torch.randperm(len(tr), generator=gen)
        sse = 0.0
        for s in range(0, len(perm), cfg.tt_batch):
            b = perm[s:s + cfg.tt_batch]
            pred = model(loo_user_x(b.numpy()), IX[ti[b]])
            loss = ((pred - tr[b]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sse += loss.item() * len(b)
        train_rmse = float(np.sqrt(sse / max(len(tr), 1)))
        v = val_rmse()
        history["train_rmse"].append(train_rmse)
        history["val_rmse"].append(v)
        log(f"[content] epoch {epoch:2d} train_rmse={train_rmse:.4f} val_rmse={v:.4f} ({time.time() - t0:.1f}s)")
        if v < best_val - 1e-5:
            best_val, bad, history["best_epoch"] = v, 0, epoch
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.tt_patience:
                log(f"[content] early stop at epoch {epoch} (best {history['best_epoch']})")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    model.export_inputs = {"feat_mean": feat_mean, "feat_std": feat_std, "item_X": item_X,
                           "user_feat_raw": user_raw}
    return model, history


# ----------------------------------------------------------------------------- export + numpy inference
def export_two_tower(model, feat_mean, feat_std, item_X, user_feat_raw, path) -> None:
    """Write models/two_tower.npz; Linear weights are stored as [in, out] so forward = x @ W + b."""
    u1, u2 = model.user_tower[0], model.user_tower[2]
    i1, i2 = model.item_tower[0], model.item_tower[2]

    def np_(t):
        return t.detach().cpu().numpy().astype(np.float32)

    tt = {
        "W_u1": np_(u1.weight).T.copy(), "b_u1": np_(u1.bias),
        "W_u2": np_(u2.weight).T.copy(), "b_u2": np_(u2.bias),
        "W_i1": np_(i1.weight).T.copy(), "b_i1": np_(i1.bias),
        "W_i2": np_(i2.weight).T.copy(), "b_i2": np_(i2.bias),
        "mu": np.array(float(model.mu), dtype=np.float32),
        "feat_mean": np.asarray(feat_mean, dtype=np.float32),
        "feat_std": np.asarray(feat_std, dtype=np.float32),
        "user_feat_raw": np.asarray(user_feat_raw, dtype=np.float32),
    }
    tt["item_vectors"] = item_vectors_np(tt, item_X)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **tt)


def load_two_tower(models_dir=MODELS_DIR) -> dict:
    """Load two_tower.npz as a dict of numpy arrays (mu as a python float)."""
    with np.load(Path(models_dir) / "two_tower.npz", allow_pickle=False) as z:
        tt = {k: z[k] for k in z.files}
    tt["mu"] = float(tt["mu"])
    return tt


def _mlp(x: np.ndarray, W1, b1, W2, b2) -> np.ndarray:
    return np.maximum(x @ W1 + b1, 0.0) @ W2 + b2


def item_vectors_np(tt: dict, item_X) -> np.ndarray:
    """Item tower forward in numpy: [I, 35] -> [I, dim]."""
    x = np.asarray(item_X, dtype=np.float32)
    return _mlp(x, tt["W_i1"], tt["b_i1"], tt["W_i2"], tt["b_i2"]).astype(np.float32)


def user_vector_np(tt: dict, user_feat_raw) -> np.ndarray:
    """Standardize raw profile(s) then user tower forward in numpy; accepts [40] or [B, 40]."""
    x = (np.asarray(user_feat_raw, dtype=np.float32) - tt["feat_mean"]) / tt["feat_std"]
    return _mlp(x, tt["W_u1"], tt["b_u1"], tt["W_u2"], tt["b_u2"]).astype(np.float32)


def score_users_tt(tt: dict, user_vecs) -> np.ndarray:
    """Predicted ratings for every item: mu + user_vecs @ item_vectors.T ([I] or [B, I])."""
    return float(tt["mu"]) + np.asarray(user_vecs) @ tt["item_vectors"].T


# ----------------------------------------------------------------------------- pipeline steps
def _dirs(ctx: dict) -> tuple[Path, Path, Path]:
    return (Path(ctx.get("models_dir", MODELS_DIR)), Path(ctx.get("processed_dir", PROCESSED_DIR)),
            Path(ctx.get("figures_dir", FIGURES_DIR)))


def train_step(cfg, ctx: dict) -> dict:
    """Train the two-tower model; write two_tower.npz and tt_history.json into models_dir."""
    from . import data as data_mod

    models_dir, processed_dir, _ = _dirs(ctx)
    data = ctx.get("data") or data_mod.load_processed(processed_dir)
    exclude = data_mod.flagged_indices(data["index_maps"])
    t0 = time.time()
    model, history = train_two_tower(data, cfg, exclude)
    models_dir.mkdir(parents=True, exist_ok=True)
    ex = model.export_inputs
    export_two_tower(model, ex["feat_mean"], ex["feat_std"], ex["item_X"], ex["user_feat_raw"],
                     models_dir / "two_tower.npz")
    (models_dir / "tt_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    best_val = history["val_rmse"][history["best_epoch"] - 1] if history["best_epoch"] else None
    print(f"[content] trained in {time.time() - t0:.1f}s, best epoch {history['best_epoch']}, val_rmse {best_val}")
    return {"best_epoch": history["best_epoch"], "val_rmse": best_val}


def _plot_history(history: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = np.arange(1, len(history["train_rmse"]) + 1)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, history["train_rmse"], marker="o", label="train RMSE")
    ax.plot(epochs, history["val_rmse"], marker="o", label="validation RMSE")
    if history.get("best_epoch"):
        ax.axvline(history["best_epoch"], color="gray", ls="--", label=f"best epoch ({history['best_epoch']})")
    ax.set_xlabel("epoch")
    ax.set_ylabel("RMSE")
    ax.set_title("Two-tower content model training")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _r4(x):
    return None if x is None else round(float(x), 4)


_RANK_KEYS = ("precision_at_10", "recall_at_10", "ndcg_at_10")
N_RANDOM_SEEDS = 10


def _random_scorer(seed: int, n_items: int):
    """score_fn for rank_eval: uniform-random scores from its own generator, bound at creation."""
    rng = np.random.default_rng(seed)
    return lambda users: rng.random((len(users), n_items))


def genre_popularity_scores(data: dict) -> np.ndarray:
    """Non-personalised item scores [I]: mean over the item's genres of genre_pop[g].

    genre_pop[g] = mean number of TRAIN ratings per warm movie having genre g. Only train ratings of
    warm movies and item genres are used (never cold ratings); items without a genre score 0.
    """
    G = np.asarray(data["item_features"]["genre_multi_hot"], dtype=np.float64)    # [I, 18]
    warm = np.asarray(data["warm_mask"], dtype=bool)
    rs = data["ratings"]
    train_i = rs.loc[rs["split"] == "train", "i"].to_numpy().astype(np.int64)
    train_i = train_i[warm[train_i]]
    n_train = np.bincount(train_i, minlength=len(warm)).astype(np.float64)      # [I], 0 for cold items
    n_warm_with_g = G[warm].sum(axis=0)
    genre_pop = np.where(n_warm_with_g > 0, (G[warm] * n_train[warm][:, None]).sum(axis=0)
                         / np.maximum(n_warm_with_g, 1), 0.0)
    n_genres = G.sum(axis=1)
    return np.where(n_genres > 0, G @ genre_pop / np.maximum(n_genres, 1), 0.0)


def cold_baselines(cfg, data: dict, models_dir=MODELS_DIR) -> dict:
    """Reference points for the two-tower cold-start numbers (spec 17.5).

    - ``user_bias``: what CF can predict for an item it never saw: clip(mu + bu[u], 1, 5) from
      cf_baseline.npz (item bias unknown -> 0), RMSE on all cold rows; ``rmse`` is None when
      cf_baseline.npz is missing.
    - ``random``: mean P/R/NDCG@10 over seeds cfg.seed .. cfg.seed+9 of uniform-random scores, with
      exactly the same eval users, cold candidates and protocol as the two-tower cold ranking;
      ``n_seeds`` records how many seeds were averaged.
    - ``genre_popularity``: P/R/NDCG@10 of the same score for every user (`genre_popularity_scores`),
      same protocol, eval users and cold candidates. It answers "how much of the two-tower's
      cold-start lift is just genre/popularity prior".
    """
    from .metrics import rank_eval, rmse_eval

    rs = data["ratings"]
    cold_rows = rs[rs["split"] == "cold"]
    bias_rmse = None
    if (Path(models_dir) / "cf_baseline.npz").exists() and len(cold_rows):
        from .cf import load_baseline
        base = load_baseline(models_dir)
        bias_rmse = rmse_eval(lambda u, i: base["mu"] + base["bu"][u], cold_rows)

    warm_mask = np.asarray(data["warm_mask"], dtype=bool)
    n_items = len(warm_mask)
    eval_users = np.asarray(data["eval_users"])
    batch = cfg.eval_rank_batch
    runs = [rank_eval(_random_scorer(seed, n_items), eval_users, data["holdout_cold"],
                      data["train_items"], ~warm_mask, TOP_K, batch)
            for seed in range(cfg.seed, cfg.seed + N_RANDOM_SEEDS)]
    n_eval = int(runs[0]["n_eval_users"])    # identical for every seed (relevance does not depend on scores)
    rand = {k: (_r4(np.mean([r[k] for r in runs])) if n_eval else None) for k in _RANK_KEYS}

    pop = genre_popularity_scores(data)
    gp = rank_eval(lambda users: np.tile(pop, (len(users), 1)), eval_users, data["holdout_cold"],
                   data["train_items"], ~warm_mask, TOP_K, batch)
    genre_pop = {k: (_r4(gp[k]) if n_eval else None) for k in _RANK_KEYS}
    return {"user_bias": {"rmse": _r4(bias_rmse)},
            "random": {**rand, "n_eval_users": n_eval, "n_seeds": N_RANDOM_SEEDS},
            "genre_popularity": {**genre_pop, "n_eval_users": int(gp["n_eval_users"])}}


def eval_step(cfg, ctx: dict) -> dict | None:
    """Warm test RMSE + P/R/NDCG@10 and cold-item RMSE + P/R/NDCG@10; figure two_tower_training.png.

    Returns {"two_tower": {...}, "cold_baselines": {...}} (both merged into metrics["rating_models"])
    or None if the model is missing. See `cold_baselines` for the cold-start reference points.
    Warm candidates = warm items not in the user's train set; cold candidates = cold items only.
    """
    from . import data as data_mod
    from .metrics import rank_eval, rmse_eval

    models_dir, _, figures_dir = _dirs(ctx)
    if not (models_dir / "two_tower.npz").exists():
        print(f"[content] {models_dir / 'two_tower.npz'} missing; skipping")
        return None
    data = ctx.get("data")
    if data is None or "eval_users" not in data:
        if data_mod.eval_step(cfg, ctx) is None:
            return None
        data = ctx["data"]

    tt = load_two_tower(models_dir)
    rs = data["ratings"]
    user_vecs = user_vector_np(tt, tt["user_feat_raw"])                       # [U, dim]

    def predict(u, i):
        return tt["mu"] + np.einsum("nd,nd->n", user_vecs[u], tt["item_vectors"][i])

    def score_fn(users):
        return score_users_tt(tt, user_vecs[users])

    batch = cfg.eval_rank_batch
    eval_users = np.asarray(data["eval_users"])
    warm_mask = np.asarray(data["warm_mask"], dtype=bool)
    warm = rank_eval(score_fn, eval_users, data["holdout_test"], data["train_items"], warm_mask, TOP_K, batch)
    cold = rank_eval(score_fn, eval_users, data["holdout_cold"], data["train_items"], ~warm_mask, TOP_K, batch)
    test_rows, cold_rows = rs[rs["split"] == "test"], rs[rs["split"] == "cold"]
    warm_rmse = rmse_eval(predict, test_rows) if len(test_rows) else None
    cold_rmse = rmse_eval(predict, cold_rows) if len(cold_rows) else None

    hist_path = models_dir / "tt_history.json"
    history = json.loads(hist_path.read_text(encoding="utf-8")) if hist_path.exists() else None
    if history:
        _plot_history(history, figures_dir / "two_tower_training.png")

    def _rank(res: dict) -> dict:   # null (not 0.0) when no eval user had a relevant item (spec 6.3)
        return {k: (_r4(res[k]) if res["n_eval_users"] else None) for k in _RANK_KEYS}

    out = {
        "rmse": _r4(warm_rmse), **_rank(warm),
        "best_epoch": int(history["best_epoch"]) if history else None,
        "cold": {"rmse": _r4(cold_rmse), **_rank(cold), "n_eval_users": int(cold["n_eval_users"])},
    }
    print(f"[content] warm: RMSE={out['rmse']} P@10={out['precision_at_10']} NDCG@10={out['ndcg_at_10']}; "
          f"cold: {out['cold']}")
    baselines = cold_baselines(cfg, data, models_dir)
    print(f"[content] cold baselines: {baselines}")
    return {"two_tower": out, "cold_baselines": baselines}
