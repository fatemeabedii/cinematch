"""Tests for cinematch.metrics with hand-computed values."""
import math

import numpy as np
import pandas as pd
import pytest

from cinematch.metrics import precision_recall_ndcg_at_k, rank_eval, rmse, rmse_eval


def test_rmse_known_value():
    assert rmse([1, 2, 3], [1, 2, 5]) == pytest.approx(math.sqrt(4 / 3))
    assert rmse([4.0], [4.0]) == 0.0


def test_rmse_eval_clips_to_rating_range():
    df = pd.DataFrame({"u": [0, 1], "i": [0, 1], "rating": [5.0, 1.0]})
    assert rmse_eval(lambda u, i: np.array([7.0, -3.0]), df) == 0.0
    assert rmse_eval(lambda u, i: np.array([4.0, 2.0]), df) == pytest.approx(1.0)


def test_prn_hand_built_three_users():
    # user 0: one relevant item at rank 2
    p, r, n = precision_recall_ndcg_at_k(np.array([1, 2, 3]), np.array([2]), k=3)
    assert (p, r) == pytest.approx((1 / 3, 1.0))
    assert n == pytest.approx(1 / math.log2(3))
    # user 1: hits at ranks 1 and 3, three relevant
    p, r, n = precision_recall_ndcg_at_k(np.array([4, 5, 6]), np.array([4, 6, 9]), k=3)
    idcg = 1 + 1 / math.log2(3) + 1 / math.log2(4)
    assert (p, r) == pytest.approx((2 / 3, 2 / 3))
    assert n == pytest.approx(1.5 / idcg)
    # user 2: no hits
    assert precision_recall_ndcg_at_k(np.array([7, 8, 9]), np.array([1]), k=3) == (0.0, 0.0, 0.0)


def test_prn_uses_only_top_k_and_idcg_min_k():
    ranked = np.arange(20)
    p, r, n = precision_recall_ndcg_at_k(ranked, np.array([0, 15]), k=10)
    assert p == pytest.approx(0.1) and r == pytest.approx(0.5)
    assert n == pytest.approx(1.0 / (1 + 1 / math.log2(3)))  # IDCG over min(10, 2) = 2 positions
    _, _, n_all = precision_recall_ndcg_at_k(ranked, np.arange(10, 30), k=10)
    assert n_all == 0.0


def test_rank_eval_masks_train_items_and_candidates():
    # 3 users x 6 items; item 5 is not a candidate (e.g. cold)
    scores = np.array([
        [9, 8, 1, 1, 1, 10],   # user 0: item 0 is a train item, item 5 non-candidate -> top-1 must be item 1
        [1, 2, 3, 4, 5, 6],    # user 1: best candidate is item 4, relevant
        [5, 4, 3, 2, 1, 0],    # user 2: no relevant held-out item -> skipped
    ], dtype=float)
    holdout = {0: {1: 5.0, 2: 2.0}, 1: {4: 4.0, 3: 3.0}, 2: {0: 3.0}}
    train_items = [np.array([0]), np.array([], dtype=int), np.array([1])]
    mask = np.array([True] * 5 + [False])
    calls = []

    def score_fn(users):
        calls.append(np.asarray(users).copy())
        return scores[users]

    out = rank_eval(score_fn, np.array([0, 1, 2]), holdout, train_items, mask, k=1, batch=2)
    assert out == {"precision_at_10": 1.0, "recall_at_10": 1.0, "ndcg_at_10": 1.0, "n_eval_users": 2}
    assert len(calls) == 2  # batched
    assert scores[0, 0] == 9  # score_fn output is not mutated in place


def test_rank_eval_train_item_never_in_top_k():
    rng = np.random.default_rng(0)
    n_users, n_items = 20, 30
    scores = rng.normal(size=(n_users, n_items))
    order = np.argsort(-scores, axis=1)
    train_items = [order[u, :5] for u in range(n_users)]  # the 5 best-scored items are train items
    # the only non-train relevant item is ranked 6th overall: masked train items -> it is top-1,
    # unmasked -> the top 5 are all train items (dropped from `rel`) and P@5 would be 0
    holdout = {u: {int(order[u, 5]): 5.0} for u in range(n_users)}
    out = rank_eval(lambda us: scores[us], np.arange(n_users), holdout, train_items,
                    np.ones(n_items, dtype=bool), k=5)
    assert out["n_eval_users"] == n_users
    assert out["precision_at_10"] == pytest.approx(1 / 5)
    assert out["recall_at_10"] == pytest.approx(1.0) and out["ndcg_at_10"] == pytest.approx(1.0)


def test_rank_eval_no_users():
    out = rank_eval(lambda us: np.zeros((len(us), 3)), np.array([0]), {}, [np.array([], int)],
                    np.ones(3, bool))
    assert out["n_eval_users"] == 0 and out["precision_at_10"] == 0.0
