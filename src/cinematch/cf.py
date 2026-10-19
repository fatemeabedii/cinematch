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

from .config import FIGURES_DIR, MODELS_DIR, PROCESSED_DIR, RELEVANT_THRESHOLD, TOP_K

TRENDING_DAYS = 90
NEW_USER_FOLD_IN_REG = 10.0  # ridge weight when folding a new user into the MF space (recommend.py imports it)


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


def rank_scores(cf: dict, P_rows, bu_rows, count, beta: float) -> np.ndarray:
    """MF ranking score: predicted rating + beta * log1p(train count)  ->  [B, I].

    Ranking by predicted rating alone collapses to the item bias, which rewards films rated 5 by a
    handful of people. The popularity prior shrinks the list toward films many people watched;
    beta is chosen on the validation split (cf_ranking.json). `score_users` stays the predicted rating.
    """
    return score_users(cf, P_rows, bu_rows) + beta * np.log1p(np.asarray(count, dtype=np.float32))[None, :]


# ----------------------------------------------------------------------------- item-item kNN
def liked_matrix(ratings_split: pd.DataFrame, n_users: int, n_items: int, exclude_users=()) -> sp.csr_matrix:
    """Binary CSR [U, I]: 1 where a TRAIN rating is >= RELEVANT_THRESHOLD (rows of `exclude_users` dropped)."""
    df = ratings_split[(ratings_split["split"] == "train") & (ratings_split["rating"] >= RELEVANT_THRESHOLD)]
    if len(exclude_users):
        df = df[~df["u"].isin(np.asarray(list(exclude_users)))]
    return sp.csr_matrix((np.ones(len(df), np.float32), (df["u"].to_numpy(), df["i"].to_numpy())),
                         shape=(n_users, n_items))


def build_item_knn(liked: sp.csr_matrix, k: int) -> sp.csr_matrix:
    """Cosine item-item similarity of the binary `liked` [U, I] matrix, top-k per row  ->  CSR [I, I] float32.

    S[i, j] = |users who liked i and j| / sqrt(|liked i| * |liked j|); the diagonal is zero and only
    the k largest positive entries of each row are kept (row i = the neighbours of item i).
    """
    liked = sp.csr_matrix(liked, dtype=np.float32)
    n_items = liked.shape[1]
    norms = np.sqrt(np.asarray(liked.sum(axis=0), dtype=np.float64).ravel())
    inv = np.divide(1.0, norms, out=np.zeros_like(norms), where=norms > 0).astype(np.float32)
    sim = (liked.T @ liked).toarray().astype(np.float32)
    sim *= inv[:, None]
    sim *= inv[None, :]
    np.fill_diagonal(sim, 0.0)
    k = int(min(k, max(n_items - 1, 1)))
    top = np.argpartition(-sim, k - 1, axis=1)[:, :k] if n_items > 1 else np.zeros((n_items, 0), np.int64)
    rows = np.repeat(np.arange(n_items), top.shape[1])
    cols = top.ravel()
    vals = sim[rows, cols]
    keep = vals > 0
    S = sp.csr_matrix((vals[keep], (rows[keep], cols[keep])), shape=(n_items, n_items), dtype=np.float32)
    S.sort_indices()
    return S


def save_item_knn(S: sp.csr_matrix, path) -> None:
    """Write item_knn.npz with the CSR arrays (data, indices, indptr, shape); loadable with numpy only."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    S = sp.csr_matrix(S)
    np.savez(path, data=S.data.astype(np.float32), indices=S.indices, indptr=S.indptr,
             shape=np.asarray(S.shape, dtype=np.int64))


def load_item_knn(models_dir=MODELS_DIR) -> sp.csr_matrix:
    """The item-kNN similarity CSR [I, I] from models_dir/item_knn.npz."""
    z = _load_npz(Path(models_dir) / "item_knn.npz", ("data", "indices", "indptr", "shape"))
    return sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(int(x) for x in z["shape"]))


def knn_scores(S: sp.csr_matrix, liked_rows: sp.csr_matrix) -> np.ndarray:
    """Item-kNN scores = liked_rows @ S  ->  dense [B, I]: each liked item votes for its neighbours."""
    return np.asarray((sp.csr_matrix(liked_rows, dtype=np.float32) @ S).toarray())


def knn_top_contributor(S: sp.csr_matrix, liked_items: np.ndarray, item: int) -> int:
    """The liked item whose similarity to `item` is largest (the "because you liked" anchor); -1 if none > 0."""
    liked_items = np.asarray(liked_items, dtype=np.int64)
    if len(liked_items) == 0:
        return -1
    sims = np.asarray(S[liked_items, int(item)].toarray()).ravel()
    best = int(np.argmax(sims))
    return int(liked_items[best]) if sims[best] > 0 else -1


def seed_liked_items(ratings: dict[int, float]) -> np.ndarray:
    """Items a new user liked: ratings >= 4, else ratings >= 3, else none (then rank by popularity)."""
    for threshold in (RELEVANT_THRESHOLD, 3):
        items = [i for i, r in ratings.items() if r >= threshold]
        if items:
            return np.asarray(items, dtype=np.int64)
    return np.zeros(0, dtype=np.int64)


def new_user_knn_scores(S: sp.csr_matrix, ratings: dict[int, float], count, gamma: float,
                        popularity) -> np.ndarray:
    """New-user item-kNN ranking scores [I]: kNN on the liked seeds + gamma*log1p(count); popularity if none liked."""
    liked = seed_liked_items(ratings)
    if len(liked) == 0:
        return np.asarray(popularity, dtype=np.float64).copy()
    row = sp.csr_matrix((np.ones(len(liked), np.float32), (np.zeros(len(liked), np.int64), liked)),
                        shape=(1, S.shape[0]))
    return knn_scores(S, row)[0].astype(np.float64) + gamma * np.log1p(np.asarray(count, dtype=np.float64))


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
    if data is None or "eval_users" not in data:  # run alone: load processed data + eval users
        if data_mod.eval_step(cfg, ctx) is None:
            data_mod.load_processed(Path(ctx.get("processed_dir", PROCESSED_DIR)))  # raises the helpful error
        data = ctx["data"]
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
    pop = popularity_scores(fit_rows, n_items)
    np.savez(models_dir / "popularity.npz", **pop)

    liked = liked_matrix(rs, n_users, n_items, exclude_users=exclude)
    ranking, S = select_ranking(cfg, data, load_cf(models_dir), liked, pop["count"])
    save_item_knn(S, models_dir / "item_knn.npz")
    (models_dir / "cf_ranking.json").write_text(json.dumps(ranking, indent=2), encoding="utf-8")
    print(f"[cf] wrote cf_baseline.npz, cf_mf.npz, popularity.npz, cf_history.json, item_knn.npz, "
          f"cf_ranking.json to {models_dir}")
    return {"best_epoch": history["best_epoch"], "best_val_rmse": min(history["val_rmse"]),
            "baseline_val_rmse": history["baseline_val_rmse"], "rank_beta": ranking["beta"],
            "knn_k": ranking["knn_k"], "knn_gamma": ranking["knn_gamma"]}


def select_ranking(cfg, data: dict, mf: dict, liked: sp.csr_matrix, count) -> tuple[dict, sp.csr_matrix]:
    """Choose the MF popularity weight beta and the item-kNN (k, gamma) by VALIDATION NDCG@10.

    Protocol = the test protocol with the val split as holdout: same eval users, warm candidates,
    train items masked, rating >= 4 relevant. Ties keep the earlier grid value. Returns the
    cf_ranking.json dict and the similarity matrix of the chosen k.
    """
    from . import data as data_mod
    from .metrics import rank_eval

    key = f"ndcg_at_{TOP_K}"
    lc = np.log1p(np.asarray(count, dtype=np.float32))[None, :]
    args = dict(eval_users=data["eval_users"], holdout=data_mod.holdout_dict(data["ratings"], "val"),
                train_items=data["train_items"], candidate_mask=data["warm_mask"], k=TOP_K, batch=cfg.eval_rank_batch)

    beta_ndcg = {}
    for beta in cfg.cf_rank_pop_betas:
        score_fn = lambda u, b=beta: rank_scores(mf, mf["P"][u], mf["bu"][u], count, b)  # noqa: E731
        beta_ndcg[float(beta)] = rank_eval(score_fn, **args)[key]
    best_beta = max(beta_ndcg, key=beta_ndcg.get)  # first maximum in grid order

    knn_ndcg, sims = {}, {}
    for k in cfg.knn_neighbors:
        sims[int(k)] = S = build_item_knn(liked, int(k))
        for gamma in cfg.knn_pop_gammas:
            score_fn = lambda u, S=S, g=gamma: knn_scores(S, liked[u]) + g * lc  # noqa: E731
            knn_ndcg[(int(k), float(gamma))] = rank_eval(score_fn, **args)[key]
    best_k, best_gamma = max(knn_ndcg, key=knn_ndcg.get)

    print(f"[cf] val NDCG@10 by beta: {_fmt_grid(beta_ndcg)} -> beta={best_beta}")
    print(f"[cf] val NDCG@10 by (k, gamma): {_fmt_grid(knn_ndcg)} -> k={best_k}, gamma={best_gamma}")
    ranking = {"beta": best_beta, "knn_k": best_k, "knn_gamma": best_gamma,
               "val_ndcg": {"beta": {str(b): round(v, 4) for b, v in beta_ndcg.items()},
                            "knn": {f"{k},{g}": round(v, 4) for (k, g), v in knn_ndcg.items()}}}
    return ranking, sims[best_k]


def _fmt_grid(grid: dict) -> str:
    return ", ".join(f"{k}: {v:.4f}" for k, v in grid.items())


def load_ranking(models_dir=MODELS_DIR) -> dict:
    """cf_ranking.json: {"beta", "knn_k", "knn_gamma", "val_ndcg"} chosen on validation by train_step."""
    return json.loads((Path(models_dir) / "cf_ranking.json").read_text(encoding="utf-8"))


def _prn(r: dict) -> dict:
    """The precision/recall/NDCG@k entries of a rank_eval result."""
    return {m: r[m] for m in (f"precision_at_{TOP_K}", f"recall_at_{TOP_K}", f"ndcg_at_{TOP_K}")}


def simulate_new_users(cfg, data: dict, mf: dict, S: sp.csr_matrix, pop, count, beta: float,
                       gamma: float) -> dict:
    """Replay eval users as brand-new app users who rated only a few well-known movies.

    For `cfg.new_user_eval_users` seeded eval users, the only known ratings are
    `cfg.new_user_seed_ratings` random TRAIN ratings on movies with >= `cfg.new_user_min_train_ratings`
    train ratings. Each strategy scores like recommend.py does for new users (MF fold-in with
    NEW_USER_FOLD_IN_REG; item-kNN on the liked seeds). Candidates = those well-known warm movies
    minus ALL of the user's train items; relevant = TEST rating >= 4. Users with too few eligible
    ratings or without a relevant candidate are skipped.
    """
    from .metrics import precision_recall_ndcg_at_k

    rng = np.random.default_rng(cfg.seed)
    count = np.asarray(count, dtype=np.float64)
    eligible = np.asarray(data["warm_mask"], dtype=bool) & (count >= cfg.new_user_min_train_ratings)
    users = np.asarray(data["eval_users"])
    if len(users) > cfg.new_user_eval_users:
        users = np.sort(rng.choice(users, cfg.new_user_eval_users, replace=False))
    rs = data["ratings"]
    tr = rs[(rs["split"] == "train") & rs["u"].isin(users)]
    by_user = {u: (g["i"].to_numpy(), g["rating"].to_numpy(np.float64)) for u, g in tr.groupby("u")}
    holdout = data["holdout_test"]
    names = ("popularity", "mf_foldin", "mf_foldin_pop", "item_knn")
    sums, n_done = {name: np.zeros(3) for name in names}, 0
    for u in users.tolist():
        items, ratings = by_user.get(u, (np.zeros(0, np.int64), np.zeros(0)))
        ok = np.flatnonzero(eligible[items])
        if len(ok) < cfg.new_user_seed_ratings:
            continue
        pick = rng.choice(ok, cfg.new_user_seed_ratings, replace=False)
        seeds = dict(zip(items[pick].tolist(), ratings[pick].tolist()))
        cand = eligible.copy()
        cand[items] = False
        rel = np.array([i for i, r in holdout.get(u, {}).items() if r >= RELEVANT_THRESHOLD and cand[i]], np.int64)
        if len(rel) == 0:
            continue
        p, bu = fold_in_user(seeds, mf, reg=NEW_USER_FOLD_IN_REG)
        scores = {"popularity": np.asarray(pop, dtype=np.float64),
                  "mf_foldin": score_users(mf, p, bu)[0],
                  "mf_foldin_pop": rank_scores(mf, p, bu, count, beta)[0],
                  "item_knn": new_user_knn_scores(S, seeds, count, gamma, pop)}
        for name, s in scores.items():
            ranked = np.argsort(-np.where(cand, s, -np.inf), kind="stable")[:TOP_K]
            sums[name] += precision_recall_ndcg_at_k(ranked, rel, TOP_K)
        n_done += 1
    strategies = {}
    for name, total in sums.items():
        m = total / max(n_done, 1)
        strategies[name] = _round({f"precision_at_{TOP_K}": float(m[0]), f"recall_at_{TOP_K}": float(m[1]),
                                   f"ndcg_at_{TOP_K}": float(m[2])})
    return {"n_users": n_done, "n_seed_ratings": int(cfg.new_user_seed_ratings), "strategies": strategies}


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
    """Test RMSE + P/R/NDCG@10 for popularity, bias baseline, MF and the ranking rules; figure cf_training.png.

    "cf_mf" ranks by predicted rating (kept as a finding: it collapses to the item bias), "cf_mf_pop"
    by rank_scores with the validation-chosen beta, "item_knn" by kNN + gamma*log1p(count);
    "new_users" is the simulate_new_users table. Returns these metrics.json "rating_models" entries,
    or None when data or CF artifacts are missing.
    """
    from . import data as data_mod
    from .metrics import rank_eval, rmse_eval

    models_dir = Path(ctx.get("models_dir", MODELS_DIR))
    figures_dir = Path(ctx.get("figures_dir", FIGURES_DIR))
    needed = [models_dir / f for f in ("cf_baseline.npz", "cf_mf.npz", "popularity.npz", "cf_history.json",
                                       "item_knn.npz", "cf_ranking.json")]
    if not all(p.exists() for p in needed):
        print(f"[cf] artifacts missing in {models_dir}; skipping eval")
        return None
    data = _get_data(cfg, ctx)
    if data is None:
        return None

    base, mf = load_baseline(models_dir), load_cf(models_dir)
    popz = _load_npz(models_dir / "popularity.npz", ("score", "count"))
    pop, count = popz["score"], popz["count"]
    S, ranking = load_item_knn(models_dir), load_ranking(models_dir)
    beta, gamma = float(ranking["beta"]), float(ranking["knn_gamma"])
    n_users = data["stats"]["n_users"]
    liked = liked_matrix(data["ratings"], n_users, S.shape[0],
                         exclude_users=data_mod.flagged_indices(data["index_maps"]))
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
    r = rank_eval(lambda u: rank_scores(mf, mf["P"][u], mf["bu"][u], count, beta), **rank_args)
    out["cf_mf_pop"] = _round({**_prn(r), "beta": beta})
    lc = np.log1p(np.asarray(count, dtype=np.float32))[None, :]
    r = rank_eval(lambda u: knn_scores(S, liked[u]) + gamma * lc, **rank_args)
    out["item_knn"] = _round({**_prn(r), "k": int(ranking["knn_k"]), "gamma": gamma})
    out["new_users"] = simulate_new_users(cfg, data, mf, S, pop, count, beta, gamma)
    _plot_history(history, figures_dir / "cf_training.png")
    print(f"[cf] eval ({r['n_eval_users']} ranked users): {out}")
    return out
