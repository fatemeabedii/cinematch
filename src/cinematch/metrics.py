"""Evaluation metrics (pure numpy): RMSE and top-k ranking metrics with binary relevance."""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np

if TYPE_CHECKING:  # annotation only; keep this module numpy-only at runtime
    import pandas as pd

from .config import RELEVANT_THRESHOLD, TOP_K


def rmse(y_true, y_pred) -> float:
    """Root mean squared error."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def precision_recall_ndcg_at_k(ranked_items: np.ndarray, relevant: np.ndarray,
                               k: int = TOP_K) -> tuple[float, float, float]:
    """P@k = hits/k, R@k = hits/n_relevant, binary NDCG@k (IDCG over min(k, n_relevant) hits).

    `ranked_items` is ordered best-first; only its first k entries are used.
    """
    top = np.asarray(ranked_items)[:k]
    relevant = np.asarray(relevant)
    if len(relevant) == 0:
        return 0.0, 0.0, 0.0
    hit = np.isin(top, relevant)
    n_hits = int(hit.sum())
    discounts = 1.0 / np.log2(np.arange(2, k + 2))  # rank 1 -> 1/log2(2)
    dcg = float(discounts[:len(top)][hit].sum())
    idcg = float(discounts[:min(k, len(relevant))].sum())
    return n_hits / k, n_hits / len(relevant), dcg / idcg


def rank_eval(score_fn: Callable[[np.ndarray], np.ndarray], eval_users: np.ndarray,
              holdout: dict[int, dict[int, float]], train_items: list[np.ndarray],
              candidate_mask: np.ndarray, k: int = TOP_K, batch: int = 256) -> dict:
    """Average P@k / R@k / NDCG@k over `eval_users` (keys precision_at_{k}, recall_at_{k}, ndcg_at_{k}).

    score_fn(user_idx_batch) -> [B, I] scores. Train items and non-candidates (~candidate_mask) are
    set to -inf before ranking. Relevant = held-out rating >= RELEVANT_THRESHOLD among candidates;
    users without any relevant candidate are skipped. Ties are broken by lower item index.
    """
    candidate_mask = np.asarray(candidate_mask, dtype=bool)
    eval_users = np.asarray(eval_users)
    sums = np.zeros(3)
    n_users = 0
    for start in range(0, len(eval_users), batch):
        users = eval_users[start:start + batch]
        scores = np.array(score_fn(users), dtype=np.float64, copy=True)
        scores[:, ~candidate_mask] = -np.inf
        for row, u in enumerate(users.tolist()):
            seen = train_items[u]
            scores[row, seen] = -np.inf
            rel = np.array([i for i, r in holdout.get(u, {}).items() if r >= RELEVANT_THRESHOLD], dtype=np.int64)
            if len(rel):
                rel = rel[candidate_mask[rel] & ~np.isin(rel, seen)]
            if len(rel) == 0:
                continue
            ranked = np.argsort(-scores[row], kind="stable")[:k]
            sums += precision_recall_ndcg_at_k(ranked, rel, k)
            n_users += 1
    means = sums / n_users if n_users else np.zeros(3)
    return {f"precision_at_{k}": float(means[0]), f"recall_at_{k}": float(means[1]),
            f"ndcg_at_{k}": float(means[2]), "n_eval_users": int(n_users)}


def rmse_eval(predict_fn: Callable[[np.ndarray, np.ndarray], np.ndarray], holdout_df: pd.DataFrame) -> float:
    """RMSE of predict_fn(u_array, i_array) clipped to [1, 5] against `holdout_df` (columns u, i, rating)."""
    u = holdout_df["u"].to_numpy()
    i = holdout_df["i"].to_numpy()
    pred = np.clip(np.asarray(predict_fn(u, i), dtype=np.float64), 1.0, 5.0)
    return rmse(holdout_df["rating"].to_numpy(), pred)
