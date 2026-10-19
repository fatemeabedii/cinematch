"""K-means taste profiles on L2-normalised CF user vectors (spec sections 5 and 9).

Users are clustered in cosine geometry (rows of `P` from `cf_mf.npz`, L2-normalised). k is chosen by
silhouette under a minimum-cluster-size rule and a readability cap (spec 17.2). Each cluster is named
by the genres its users over-rate in percentage points (spec 17.3) and gets a top-50 recommendation
list.

Numpy/pandas only at import time (sklearn and matplotlib are imported lazily; torch is never used).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from cinematch.config import FIGURES_DIR, GENRES, MODELS_DIR, PROCESSED_DIR, RELEVANT_THRESHOLD

SILHOUETTE_SAMPLE = 3000
FALLBACK_K = 4             # used when no k satisfies the min-cluster-size rule
MAINSTREAM_PP = 2.0        # top genre over-represented by less than this (percentage points) -> "Mainstream"
N_TOP_GENRES = 5           # genres stored per cluster in clusters.json
N_RECS = 50


def _log(msg: str) -> None:
    print(f"[clustering] {msg}")


def _l2_normalise(X: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation; all-zero rows stay zero."""
    X = np.asarray(X, dtype=np.float64)
    norms = np.linalg.norm(X, axis=-1, keepdims=True)
    return X / np.maximum(norms, 1e-12)


def _silhouette_sample_idx(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=min(SILHOUETTE_SAMPLE, n), replace=False))


def _silhouette(X: np.ndarray, labels: np.ndarray, sample_idx: np.ndarray) -> float:
    from sklearn.metrics import silhouette_score

    lab = labels[sample_idx]
    if len(np.unique(lab)) < 2:
        return -1.0
    return float(silhouette_score(X[sample_idx], lab))


def _kmeans(X: np.ndarray, k: int, seed: int):
    from sklearn.cluster import KMeans

    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)


# ----------------------------------------------------------------------------- k selection
def fit_kmeans_range(X: np.ndarray, k_min: int, k_max: int, seed: int) -> dict:
    """Fit K-means for every k in [k_min, k_max] and record inertia, silhouette and smallest cluster share.

    Silhouette is computed on a seeded sample of at most 3,000 rows (the same rows for every k).
    Returns JSON-ready lists: {"ks", "inertia", "silhouette", "min_frac"}.
    """
    X = np.asarray(X, dtype=np.float64)
    n = len(X)
    sample_idx = _silhouette_sample_idx(n, seed)
    scan = {"ks": [], "inertia": [], "silhouette": [], "min_frac": []}
    for k in range(int(k_min), min(int(k_max), n - 1) + 1):
        km = _kmeans(X, k, seed)
        sizes = np.bincount(km.labels_, minlength=k)
        scan["ks"].append(k)
        scan["inertia"].append(float(km.inertia_))
        scan["silhouette"].append(_silhouette(X, km.labels_, sample_idx))
        scan["min_frac"].append(float(sizes.min() / n))
    return scan


def choose_k(scan: dict, k_cap: int, min_frac: float) -> int:
    """Pick k from a `fit_kmeans_range` scan (spec 17.2).

    Candidates are k <= k_cap whose smallest cluster holds >= min_frac of users; take the one with
    the highest silhouette, exact ties -> smaller k. No candidate -> k = 4. (The scan itself starts
    at cfg.k_min.)
    """
    cands = [(k, s) for k, s, f in zip(scan["ks"], scan["silhouette"], scan["min_frac"])
             if k <= k_cap and f >= min_frac and np.isfinite(s)]
    if not cands:
        return FALLBACK_K
    return int(min(cands, key=lambda ks: (-ks[1], ks[0]))[0])


# ----------------------------------------------------------------------------- naming
def _genre_shares(items: np.ndarray, genre_multi_hot: np.ndarray) -> np.ndarray:
    """Fraction of the ratings in `items` (item indices, one per rating) whose movie has each genre."""
    if len(items) == 0:
        return np.zeros(genre_multi_hot.shape[1])
    return genre_multi_hot[items].sum(axis=0) / len(items)


def _make_name(genres: list[str], scores: list[float], n: int) -> str:
    """"A & B fans" (n=2) or "A, B & C fans" (n=3) from the top-scored genres; "Mainstream" when the
    best genre is over-represented by less than MAINSTREAM_PP percentage points.

    Follows spec 17.3 literally: the 2nd/3rd genre is taken by rank even when its own score is <= 0
    (e.g. "Comedy & Action fans" where Action is merely the first 0.0 pp genre in ML-1M order).
    That edge case does not occur on the real data (all named clusters have G2 > 0)."""
    if scores[0] < MAINSTREAM_PP:
        return "Mainstream"
    picked = genres[:n]
    return f"{', '.join(picked[:-1])} & {picked[-1]} fans"


def name_clusters(labels: np.ndarray, ratings_split: pd.DataFrame, movies: pd.DataFrame | None,
                  genre_multi_hot: np.ndarray | None, *, n_clusters: int | None = None) -> list[dict]:
    """Name each cluster by the genres its users over-rate, in percentage points (spec 17.3).

    Uses ALL train ratings >= 4 in `ratings_split` (pass only the fit users' rows). For cluster c,
    share_c[g] = fraction of its users' liked ratings whose movie has genre g; share_all[g] = the same
    over every user in `ratings_split`; score = 100 * (share_c - share_all). Name = "G1 & G2 fans" for
    the top-2 genres by score (ties -> ML-1M genre order), or "Mainstream" if the top score < 2.0 pp.
    Duplicate names get the 3rd genre ("A, B & C fans"), then a " #<id>" suffix, so names are unique.
    `labels[u]` is the cluster of user index u (`size` counts all of them). `genre_multi_hot` is
    [I, 18] (built from `movies["genres"]` ordered by `i` when None); `n_clusters` defaults to
    labels.max() + 1.
    Returns per cluster {"cluster_id", "name", "top_genres": [[genre, pp_score], ...] (top 5,
    1 decimal), "size"}.
    """
    labels = np.asarray(labels)
    if genre_multi_hot is None:
        from cinematch.data import genre_multi_hot as _gmh
        genre_multi_hot = _gmh(movies.sort_values("i")["genres"])
    genre_multi_hot = np.asarray(genre_multi_hot, dtype=np.float64)
    k = int(n_clusters) if n_clusters is not None else (int(labels.max()) + 1 if len(labels) else 0)

    liked = ratings_split[(ratings_split["split"] == "train")
                          & (ratings_split["rating"] >= RELEVANT_THRESHOLD)]
    u = liked["u"].to_numpy(np.int64)
    i = liked["i"].to_numpy(np.int64)
    row_cluster = labels[u]
    share_all = _genre_shares(i, genre_multi_hot)

    sizes = np.bincount(labels, minlength=k)
    base, ranked = [], []
    for c in range(k):
        score = 100.0 * (_genre_shares(i[row_cluster == c], genre_multi_hot) - share_all)
        order = np.lexsort((np.arange(len(score)), -score))           # score desc, genre order asc
        genres = [GENRES[j] for j in order]
        scores = [float(score[j]) for j in order]
        ranked.append((genres, scores))
        base.append(_make_name(genres, scores, 2))

    names = list(base)
    for c in range(k):
        if base.count(base[c]) > 1 and base[c] != "Mainstream":
            names[c] = _make_name(*ranked[c], 3)
    clashing = [names.count(n) > 1 for n in names]
    names = [f"{n} #{c}" if clash else n for c, (n, clash) in enumerate(zip(names, clashing))]

    return [{"cluster_id": c, "name": names[c],
             "top_genres": [[g, round(sc, 1) + 0.0] for g, sc in zip(ranked[c][0][:N_TOP_GENRES],
                                                                ranked[c][1][:N_TOP_GENRES])],
             "size": int(sizes[c])}
            for c in range(k)]


# ----------------------------------------------------------------------------- recs + assignment
def cluster_recs(centroid_raw: np.ndarray, cf: dict, popularity: dict, warm_mask: np.ndarray,
                 k: int = 50) -> np.ndarray:
    """Top-k warm item indices for a cluster: mu + bi + centroid_raw @ Q.T + 0.1 * log1p(count).

    `centroid_raw` is the mean of the members' unnormalised P rows (same scale as Q); `popularity`
    needs key "count" (train ratings per item). Ties are broken by lower item index.
    """
    scores = (float(cf["mu"]) + np.asarray(cf["bi"], dtype=np.float64)
              + np.asarray(cf["Q"], dtype=np.float64) @ np.asarray(centroid_raw, dtype=np.float64)
              + 0.1 * np.log1p(np.asarray(popularity["count"], dtype=np.float64)))
    scores[~np.asarray(warm_mask, dtype=bool)] = -np.inf
    order = np.lexsort((np.arange(len(scores)), -scores))
    order = order[np.isfinite(scores[order])]
    return order[:k].astype(np.int64)


def assign_cluster(clusters: dict, user_vec: np.ndarray) -> int:
    """Nearest centroid by cosine similarity after L2-normalising `user_vec` (a raw CF user vector)."""
    x = _l2_normalise(np.asarray(user_vec, dtype=np.float64).ravel())
    C = _l2_normalise(np.asarray(clusters["centroids"], dtype=np.float64))
    return int(np.argmax(C @ x))


def _assign_many(C: np.ndarray, Xn: np.ndarray) -> np.ndarray:
    """Vectorised `assign_cluster` for already-normalised rows."""
    return np.argmax(Xn @ _l2_normalise(C).T, axis=1).astype(np.int32)


# ----------------------------------------------------------------------------- pipeline steps
def _dirs(ctx: dict) -> tuple[Path, Path, Path]:
    return (Path(ctx.get("models_dir", MODELS_DIR)), Path(ctx.get("processed_dir", PROCESSED_DIR)),
            Path(ctx.get("figures_dir", FIGURES_DIR)))


def _load_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _metrics(meta: dict) -> dict:
    return {"k": int(meta["k"]), "silhouette": round(float(meta["silhouette"]), 4),
            "names": list(meta["names"]), "sizes": [int(s) for s in meta["sizes"]]}


def train_step(cfg, ctx: dict) -> dict:
    """Scan k, fit the final K-means, name clusters, build recs and PCA; write clusters.npz/.json.

    Fit set = non-flagged users with >= 1 train rating. K-means is fit on them; then EVERY user is
    labelled by nearest centroid (cosine, exactly as `assign_cluster`), and centroids_raw, silhouette
    and names use those labels on the fit users. `sizes` count ALL users' labels. Returns the
    metrics.json "clustering" dict.
    """
    from sklearn.decomposition import PCA

    from cinematch.data import flagged_indices

    models_dir, processed_dir, _ = _dirs(ctx)
    data = ctx.get("data")
    if data is None:
        from cinematch.data import load_processed
        data = load_processed(processed_dir)
    cf = _load_npz(models_dir / "cf_mf.npz")
    popularity = _load_npz(models_dir / "popularity.npz")
    rs = data["ratings"]
    P = np.asarray(cf["P"], dtype=np.float64)
    n_users = P.shape[0]

    n_train = np.bincount(rs.loc[rs["split"] == "train", "u"].to_numpy(np.int64), minlength=n_users)
    fit_idx = np.setdiff1d(np.flatnonzero(n_train > 0), flagged_indices(data["index_maps"]))
    Xn_all = _l2_normalise(P)
    X = Xn_all[fit_idx]
    _log(f"clustering {len(fit_idx)} of {n_users} users (dim {P.shape[1]})")

    scan = fit_kmeans_range(X, cfg.k_min, cfg.k_max, cfg.seed)
    k = choose_k(scan, cfg.k_cap, cfg.k_min_cluster_frac)
    km = _kmeans(X, k, cfg.seed)
    # One assignment rule for EVERY user (fit users included): cosine to the stored (float32)
    # centroids, exactly as `assign_cluster`. K-means' own labels_ use Euclidean distance to the
    # unnormalised centres and disagree on boundary users, so they are not kept.
    centroids = km.cluster_centers_.astype(np.float32).astype(np.float64)
    labels = _assign_many(centroids, Xn_all)
    fit_labels = labels[fit_idx]
    silhouette = _silhouette(X, fit_labels, _silhouette_sample_idx(len(X), cfg.seed))
    scan_sil = scan["silhouette"][scan["ks"].index(k)] if k in scan["ks"] else None
    P_fit = P[fit_idx]
    centroids_raw = np.stack([P_fit[fit_labels == c].mean(axis=0) if np.any(fit_labels == c)
                              else P_fit[km.labels_ == c].mean(axis=0) for c in range(k)])

    genre = data["item_features"]["genre_multi_hot"]
    rs_fit = rs[rs["u"].isin(fit_idx)]                  # names describe the non-flagged fit users
    info = name_clusters(labels, rs_fit, data["movies"], genre, n_clusters=k)
    warm_mask = data.get("warm_mask")
    if warm_mask is None:
        warm_mask = ~np.asarray(data["item_features"]["is_cold"], dtype=bool)
    recs = {str(c): [int(x) for x in cluster_recs(centroids_raw[c], cf, popularity, warm_mask, N_RECS)]
            for c in range(k)}

    pca = PCA(n_components=2, random_state=cfg.seed).fit(X)
    pca_2d = pca.transform(Xn_all)

    models_dir.mkdir(parents=True, exist_ok=True)
    np.savez(models_dir / "clusters.npz", centroids=centroids.astype(np.float32),
             centroids_raw=centroids_raw.astype(np.float32), labels=labels.astype(np.int32),
             pca_2d=pca_2d.astype(np.float32), pca_components=pca.components_.astype(np.float32),
             pca_mean=pca.mean_.astype(np.float32))
    meta = {"k": int(k), "silhouette": float(silhouette), "names": [d["name"] for d in info],
            "sizes": [d["size"] for d in info], "top_genres": [d["top_genres"] for d in info],
            "recs": recs, "scan": scan, "n_fit_users": int(len(fit_idx)),
            # Scan value at the chosen k (K-means' own Euclidean labels); "silhouette" above is on the
            # final cosine-reassigned labels, so the two differ slightly. Both are drawn in silhouette.png.
            "silhouette_scan_at_k": None if scan_sil is None else float(scan_sil)}
    (models_dir / "clusters.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    _log(f"k={k} silhouette={silhouette:.4f} (scan at k: {scan_sil}) names={meta['names']} sizes={meta['sizes']}")
    return _metrics(meta)


def load_clusters(models_dir: Path = MODELS_DIR) -> dict:
    """clusters.npz arrays and clusters.json fields merged into one dict (numpy only)."""
    models_dir = Path(models_dir)
    out = _load_npz(models_dir / "clusters.npz")
    out.update(json.loads((models_dir / "clusters.json").read_text(encoding="utf-8")))
    return out


def _plot_scan(plt, scan: dict, k: int, key: str, ylabel: str, title: str, path: Path,
               final: float | None = None) -> None:
    """Scan curve with the chosen k circled. `final` (silhouette only) adds the reported value on the
    final cosine-reassigned labels, which differs slightly from the scan value at k."""
    ks, ys = scan["ks"], scan[key]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ks, ys, marker="o", label="scan (K-means labels)" if final is not None else None)
    if k in ks:
        ax.scatter([k], [ys[ks.index(k)]], s=160, facecolors="none", edgecolors="red", linewidths=2,
                   label=f"chosen k = {k} (scan {ys[ks.index(k)]:.4f})" if final is not None
                   else f"chosen k = {k}", zorder=3)
    if final is not None:
        ax.scatter([k], [final], s=90, marker="*", color="black", zorder=4,
                   label=f"reported (final labels) {final:.4f}")
    if k in ks or final is not None:
        ax.legend()
    ax.set_xlabel("k (number of clusters)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(ks)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_pca(plt, cl: dict, path: Path, seed: int) -> None:
    xy, labels = cl["pca_2d"], cl["labels"]
    idx = np.random.default_rng(seed).choice(len(xy), size=min(SILHOUETTE_SAMPLE, len(xy)), replace=False)
    cent = (np.asarray(cl["centroids"]) - cl["pca_mean"]) @ np.asarray(cl["pca_components"]).T
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for c, name in enumerate(cl["names"]):
        m = idx[labels[idx] == c]
        ax.scatter(xy[m, 0], xy[m, 1], s=6, alpha=0.5, color=cmap(c % 10), label=f"{name} ({cl['sizes'][c]})")
        ax.scatter(cent[c, 0], cent[c, 1], s=180, marker="X", color=cmap(c % 10), edgecolors="black")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title("Taste profiles: K-means on normalised CF user vectors (PCA view)")
    ax.legend(fontsize=7, markerscale=2, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def eval_step(cfg, ctx: dict) -> dict | None:
    """Read clusters.npz/.json only; write elbow.png, silhouette.png, clusters_pca.png.

    Returns the metrics.json "clustering" dict {k, silhouette, names, sizes}, or None if the cluster
    artifacts are missing.
    """
    models_dir, _, figures_dir = _dirs(ctx)
    try:
        cl = load_clusters(models_dir)
    except FileNotFoundError:
        _log(f"clusters.npz/json missing in {models_dir}; skipping")
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir.mkdir(parents=True, exist_ok=True)
    k = int(cl["k"])
    _plot_scan(plt, cl["scan"], k, "inertia", "inertia (within-cluster SSE)",
               "Elbow plot (silhouette decides k)", figures_dir / "elbow.png")
    _plot_scan(plt, cl["scan"], k, "silhouette", "silhouette score (3,000-user sample)",
               "Silhouette vs k", figures_dir / "silhouette.png", final=float(cl["silhouette"]))
    _plot_pca(plt, cl, figures_dir / "clusters_pca.png", cfg.seed)
    return _metrics(cl)
