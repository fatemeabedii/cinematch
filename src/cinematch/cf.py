"""Collaborative filtering: bias baseline (numpy), matrix factorization (PyTorch) and popularity.

Torch is imported lazily inside functions (spec section 3), so `import cinematch.cf` stays light.
Artifacts (all `.npz`, loadable with numpy only): cf_baseline.npz, cf_mf.npz, popularity.npz,
plus cf_history.json with the MF learning curves.
"""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .config import FIGURES_DIR, MODELS_DIR, PROCESSED_DIR, TOP_K

TRENDING_DAYS = 90


def _torch():
    import torch
    return torch


# ----------------------------------------------------------------------------- bias baseline
class BiasBaseline:
    """r_hat(u, i) = mu + bu[u] + bi[i], biases fitted by alternating closed-form ridge updates."""

    def __init__(self):
        self.mu = 0.0
        self.bu = np.zeros(0, dtype=np.float32)
        self.bi = np.zeros(0, dtype=np.float32)

    def fit(self, R_train: sp.csr_matrix, reg: float = 10.0, iters: int = 10) -> "BiasBaseline":
        """Fit on a sparse [U, I] rating matrix (stored entries = observed ratings)."""
        R = sp.coo_matrix(R_train)
        u, i, r = R.row, R.col, R.data.astype(np.float64)
        n_users, n_items = R.shape
        self.mu = float(r.mean()) if len(r) else 0.0
        bu = np.zeros(n_users)
        bi = np.zeros(n_items)
        n_u = np.bincount(u, minlength=n_users)
        n_i = np.bincount(i, minlength=n_items)
        for _ in range(iters):
            bi = np.bincount(i, weights=r - self.mu - bu[u], minlength=n_items) / (reg + n_i)
            bu = np.bincount(u, weights=r - self.mu - bi[i], minlength=n_users) / (reg + n_u)
        self.bu, self.bi = bu.astype(np.float32), bi.astype(np.float32)
        return self

    def predict(self, u, i) -> np.ndarray:
        """Predicted ratings for paired index arrays u, i (unclipped)."""
        return self.mu + self.bu[np.asarray(u)] + self.bi[np.asarray(i)]

    def score_all(self, u: np.ndarray) -> np.ndarray:
        """Scores of every item for each user in `u`: [B, I]."""
        u = np.atleast_1d(np.asarray(u))
        return self.mu + self.bu[u][:, None] + self.bi[None, :]


# ----------------------------------------------------------------------------- matrix factorization
def build_mf(n_users: int, n_items: int, dim: int, mu: float):
    """MF module: forward(u, i) = mu + bu[u] + bi[i] + <P[u], Q[i]>; all parameters init N(0, 0.05)."""
    torch = _torch()
    nn = torch.nn

    class MF(nn.Module):
        def __init__(self):
            super().__init__()
            self.user_emb = nn.Embedding(n_users, dim)
            self.item_emb = nn.Embedding(n_items, dim)
            self.user_bias = nn.Embedding(n_users, 1)
            self.item_bias = nn.Embedding(n_items, 1)
            for emb in (self.user_emb, self.item_emb, self.user_bias, self.item_bias):
                nn.init.normal_(emb.weight, 0.0, 0.05)
            self.register_buffer("mu", torch.tensor(float(mu)))

        def forward(self, u, i):
            p, q = self.user_emb(u), self.item_emb(i)
            bu, bi = self.user_bias(u).squeeze(-1), self.item_bias(i).squeeze(-1)
            return self.mu + bu + bi + (p * q).sum(-1)

        def penalty(self, u, i):
            """Per-sample L2 of the parameters touched by the batch: |p|^2 + |q|^2 + bu^2 + bi^2."""
            return ((self.user_emb(u) ** 2).sum(-1) + (self.item_emb(i) ** 2).sum(-1)
                    + self.user_bias(u).squeeze(-1) ** 2 + self.item_bias(i).squeeze(-1) ** 2)

    return MF()


def _split_arrays(ratings: pd.DataFrame, split: str, exclude_users=()) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = ratings[ratings["split"] == split]
    if len(exclude_users):
        df = df[~df["u"].isin(np.asarray(list(exclude_users)))]
    return (df["u"].to_numpy(np.int64, copy=True), df["i"].to_numpy(np.int64, copy=True),
            df["rating"].to_numpy(np.float32, copy=True))


def train_mf(data: dict, cfg, exclude_users=(), log=print):
    """Train MF with mini-batch Adam on TRAIN rows (excluded users dropped); early stop on val RMSE.

    loss = MSE + cf_reg * mean(|p|^2 + |q|^2 + bu^2 + bi^2). Validation RMSE uses predictions
    clipped to [1, 5] on all val rows. The best-epoch weights are restored before returning.
    Returns (model, history) with history = {"train_rmse", "val_rmse", "best_epoch"} (1-based epoch).
    """
    torch = _torch()
    torch.manual_seed(cfg.seed)
    ratings = data["ratings"]
    n_users, n_items = data["stats"]["n_users"], data["stats"]["n_items"]
    tu, ti, tr = (torch.from_numpy(a) for a in _split_arrays(ratings, "train", exclude_users))
    vu, vi, vr = (torch.from_numpy(a) for a in _split_arrays(ratings, "val"))
    model = build_mf(n_users, n_items, cfg.cf_dim, float(tr.mean()))
    opt = torch.optim.Adam(model.parameters(), lr=cfg.cf_lr)
    gen = torch.Generator().manual_seed(cfg.seed)

    history = {"train_rmse": [], "val_rmse": [], "best_epoch": 0}
    best_val, best_state, bad = float("inf"), None, 0
    for epoch in range(1, cfg.cf_epochs + 1):
        t0 = time.time()
        model.train()
        perm = torch.randperm(len(tr), generator=gen)
        sse = 0.0
        for start in range(0, len(perm), cfg.cf_batch):
            idx = perm[start:start + cfg.cf_batch]
            u, i, r = tu[idx], ti[idx], tr[idx]
            pred = model(u, i)
            err = ((pred - r) ** 2)
            loss = err.mean() + cfg.cf_reg * model.penalty(u, i).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sse += float(err.detach().sum())
        train_rmse = (sse / max(len(tr), 1)) ** 0.5
        model.eval()
        with torch.no_grad():
            val_rmse = float(torch.sqrt(((model(vu, vi).clamp(1, 5) - vr) ** 2).mean())) if len(vr) else train_rmse
        history["train_rmse"].append(round(train_rmse, 5))
        history["val_rmse"].append(round(val_rmse, 5))
        log(f"[cf] epoch {epoch:2d} train_rmse={train_rmse:.4f} val_rmse={val_rmse:.4f} ({time.time() - t0:.1f}s)")
        if val_rmse < best_val - 1e-5:
            best_val, bad, history["best_epoch"] = val_rmse, 0, epoch
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= cfg.cf_patience:
                log(f"[cf] early stop at epoch {epoch} (best {history['best_epoch']}, val_rmse={best_val:.4f})")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


# ----------------------------------------------------------------------------- export / load
def export_mf(model, path) -> None:
    """Write cf_mf.npz: P[U,d], Q[I,d], bu[U], bi[I], mu (0-d), all float32."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path,
             P=model.user_emb.weight.detach().cpu().numpy().astype(np.float32),
             Q=model.item_emb.weight.detach().cpu().numpy().astype(np.float32),
             bu=model.user_bias.weight.detach().cpu().numpy().ravel().astype(np.float32),
             bi=model.item_bias.weight.detach().cpu().numpy().ravel().astype(np.float32),
             mu=np.array(float(model.mu), dtype=np.float32))


def export_baseline(model: BiasBaseline, path) -> None:
    """Write cf_baseline.npz: mu (0-d), bu[U], bi[I]."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, mu=np.array(model.mu, dtype=np.float32), bu=model.bu, bi=model.bi)


def _load_npz(path: Path, keys) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in keys}


def load_cf(models_dir=MODELS_DIR) -> dict:
    """{"P","Q","bu","bi","mu"} from cf_mf.npz (numpy only; mu is a float)."""
    out = _load_npz(Path(models_dir) / "cf_mf.npz", ("P", "Q", "bu", "bi", "mu"))
    out["mu"] = float(out["mu"])
    return out


def load_baseline(models_dir=MODELS_DIR) -> dict:
    """{"mu","bu","bi"} from cf_baseline.npz (mu is a float)."""
    out = _load_npz(Path(models_dir) / "cf_baseline.npz", ("mu", "bu", "bi"))
    out["mu"] = float(out["mu"])
    return out


# ----------------------------------------------------------------------------- popularity
def popularity_scores(ratings_split: pd.DataFrame, n_items: int) -> dict[str, np.ndarray]:
    """Popularity from TRAIN rows: score = count * max(mean - 2.5, 0); count; trending = count in the
    last 90 days of train timestamps. Items without train rows (incl. cold items) get 0."""
    tr = ratings_split[ratings_split["split"] == "train"]
    i = tr["i"].to_numpy()
    r = tr["rating"].to_numpy(np.float64)
    count = np.bincount(i, minlength=n_items).astype(np.float64)
    total = np.bincount(i, weights=r, minlength=n_items)
    mean = np.divide(total, count, out=np.zeros(n_items), where=count > 0)
    score = count * np.maximum(mean - 2.5, 0.0)
    ts = tr["timestamp"].to_numpy()
    recent = ts >= (ts.max() - TRENDING_DAYS * 86400) if len(ts) else np.zeros(0, bool)
    trending = np.bincount(i[recent], minlength=n_items).astype(np.float64)
    return {"score": score.astype(np.float32), "count": count.astype(np.float32),
            "trending": trending.astype(np.float32)}


# ----------------------------------------------------------------------------- inference helpers
def fold_in_user(ratings: dict[int, float], cf: dict, reg: float = 1.0) -> tuple[np.ndarray, float]:
    """Vector + bias for a new user from {item index: rating}, with item vectors frozen.

    p = (Qs^T Qs + reg I)^-1 Qs^T (r - mu - bi_s); bu = mean residual after p.
    """
    d = cf["Q"].shape[1]
    if not ratings:
        return np.zeros(d, dtype=np.float32), 0.0
    items = np.fromiter(ratings.keys(), dtype=np.int64)
    r = np.fromiter(ratings.values(), dtype=np.float64)
    Qs = cf["Q"][items].astype(np.float64)
    resid = r - cf["mu"] - cf["bi"][items]
    p = np.linalg.solve(Qs.T @ Qs + reg * np.eye(d), Qs.T @ resid)
    bu = float(np.mean(resid - Qs @ p))
    return p.astype(np.float32), bu


def score_users(cf: dict, P_rows, bu_rows) -> np.ndarray:
    """mu + bu[:, None] + bi[None, :] + P_rows @ Q.T  ->  [B, I] (1-D inputs give [1, I])."""
    P_rows = np.atleast_2d(np.asarray(P_rows, dtype=np.float32))
    bu_rows = np.atleast_1d(np.asarray(bu_rows, dtype=np.float32))
    return cf["mu"] + bu_rows[:, None] + cf["bi"][None, :] + P_rows @ cf["Q"].T


# ----------------------------------------------------------------------------- pipeline steps
def _get_data(cfg, ctx: dict) -> dict | None:
    """ctx["data"], loading processed data + eval users via data.eval_step when run alone."""
    if ctx.get("data") is None:
        from . import data as data_mod
        if data_mod.eval_step(cfg, ctx) is None:
            return None
    return ctx["data"]


def train_step(cfg, ctx: dict) -> dict:
    """Fit bias baseline, MF and popularity on train rows without flagged users; write artifacts."""
    from . import data as data_mod
    from .metrics import rmse_eval

    models_dir = Path(ctx.get("models_dir", MODELS_DIR))
    models_dir.mkdir(parents=True, exist_ok=True)
    data = ctx.get("data")
    if data is None:
        data = data_mod.load_processed(Path(ctx.get("processed_dir", PROCESSED_DIR)))
    rs = data["ratings"]
    n_users, n_items = data["stats"]["n_users"], data["stats"]["n_items"]
    exclude = data_mod.flagged_indices(data["index_maps"])

    R = data_mod.train_matrix(rs, n_users, n_items, "train", exclude_users=exclude)
    baseline = BiasBaseline().fit(R, reg=cfg.bias_reg, iters=cfg.bias_iters)
    val_df = rs[rs["split"] == "val"]
    baseline_val = rmse_eval(baseline.predict, val_df) if len(val_df) else None
    print(f"[cf] bias baseline val_rmse={baseline_val}")
    export_baseline(baseline, models_dir / "cf_baseline.npz")

    model, history = train_mf(data, cfg, exclude_users=exclude)
    export_mf(model, models_dir / "cf_mf.npz")
    history["baseline_val_rmse"] = None if baseline_val is None else round(baseline_val, 5)
    (models_dir / "cf_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    fit_rows = rs[~rs["u"].isin(exclude)] if len(exclude) else rs  # flagged users do not vote either
    np.savez(models_dir / "popularity.npz", **popularity_scores(fit_rows, n_items))
    print(f"[cf] wrote cf_baseline.npz, cf_mf.npz, popularity.npz, cf_history.json to {models_dir}")
    return {"best_epoch": history["best_epoch"], "best_val_rmse": min(history["val_rmse"]),
            "baseline_val_rmse": history["baseline_val_rmse"]}


def _round(d: dict) -> dict:
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}


def _plot_history(history: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = np.arange(1, len(history["train_rmse"]) + 1)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, history["train_rmse"], marker="o", label="MF train RMSE")
    ax.plot(epochs, history["val_rmse"], marker="o", label="MF validation RMSE")
    if history.get("baseline_val_rmse") is not None:
        ax.axhline(history["baseline_val_rmse"], color="gray", ls="--", label="bias baseline (val)")
    if history.get("best_epoch"):
        ax.axvline(history["best_epoch"], color="green", ls=":", label=f"best epoch {history['best_epoch']}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("RMSE")
    ax.set_title("Matrix factorization training")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def eval_step(cfg, ctx: dict) -> dict | None:
    """Test RMSE + P/R/NDCG@10 for popularity, bias baseline and MF; figure cf_training.png.

    Returns {"popularity", "bias_baseline", "cf_mf"} (metrics.json "rating_models" entries), or None
    when data or CF artifacts are missing.
    """
    from .metrics import rank_eval, rmse_eval

    models_dir = Path(ctx.get("models_dir", MODELS_DIR))
    figures_dir = Path(ctx.get("figures_dir", FIGURES_DIR))
    needed = [models_dir / f for f in ("cf_baseline.npz", "cf_mf.npz", "popularity.npz", "cf_history.json")]
    if not all(p.exists() for p in needed):
        print(f"[cf] artifacts missing in {models_dir}; skipping eval")
        return None
    data = _get_data(cfg, ctx)
    if data is None:
        return None

    base, mf = load_baseline(models_dir), load_cf(models_dir)
    pop = _load_npz(models_dir / "popularity.npz", ("score",))["score"]
    history = json.loads((models_dir / "cf_history.json").read_text(encoding="utf-8"))
    rs = data["ratings"]
    test_df = rs[rs["split"] == "test"]
    rank_args = dict(eval_users=data["eval_users"], holdout=data["holdout_test"], train_items=data["train_items"],
                     candidate_mask=data["warm_mask"], k=TOP_K, batch=cfg.eval_rank_batch)

    def pop_scores(users):
        return np.broadcast_to(pop, (len(users), len(pop)))

    def base_scores(users):
        return base["mu"] + base["bu"][users][:, None] + base["bi"][None, :]

    def mf_scores(users):
        return score_users(mf, mf["P"][users], mf["bu"][users])

    def mf_predict(u, i):
        return mf["mu"] + mf["bu"][u] + mf["bi"][i] + np.einsum("nd,nd->n", mf["P"][u], mf["Q"][i])

    out = {}
    r = rank_eval(pop_scores, **rank_args)
    out["popularity"] = _round({"rmse": None, "precision_at_10": r["precision_at_10"],
                                "recall_at_10": r["recall_at_10"], "ndcg_at_10": r["ndcg_at_10"]})
    r = rank_eval(base_scores, **rank_args)
    out["bias_baseline"] = _round({"rmse": rmse_eval(lambda u, i: base["mu"] + base["bu"][u] + base["bi"][i], test_df),
                                   "precision_at_10": r["precision_at_10"], "recall_at_10": r["recall_at_10"],
                                   "ndcg_at_10": r["ndcg_at_10"]})
    r = rank_eval(mf_scores, **rank_args)
    out["cf_mf"] = _round({"rmse": rmse_eval(mf_predict, test_df), "precision_at_10": r["precision_at_10"],
                           "recall_at_10": r["recall_at_10"], "ndcg_at_10": r["ndcg_at_10"],
                           "best_epoch": int(history["best_epoch"]), "dim": int(mf["P"].shape[1])})
    _plot_history(history, figures_dir / "cf_training.png")
    print(f"[cf] eval ({r['n_eval_users']} ranked users): {out}")
    return out
