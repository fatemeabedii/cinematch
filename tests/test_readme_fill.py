"""Tests for metrics assembly, README rendering and the number-provenance rule."""
import json
import re
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from cinematch.readme_fill import (
    MARKER_END,
    MARKER_START,
    assemble_metrics,
    fmt,
    render_results_markdown,
    split_readme,
    update_readme,
)

ROOT = Path(__file__).resolve().parents[1]
FOUR_DECIMALS = re.compile(r"\d\.\d{4}")


def _cfg(fast=False):
    return types.SimpleNamespace(seed=42, fast=fast, cold_item_frac=0.06, cold_item_min_ratings=20,
                                 val_frac=0.10, test_frac=0.10)


def _rank(rmse=None):
    return {"rmse": rmse, "precision_at_10": 0.1234, "recall_at_10": 0.05, "ndcg_at_10": 0.15}


def _full_outputs():
    """Fake eval_step outputs for every step, in the shapes the pipeline writes (DQN not shipped)."""
    return {
        "data": {"n_users": 6040, "n_items": 3883, "n_cold_items": 120, "n_flagged_users": 30,
                 "n_train": 790000, "n_val": 98000, "n_test": 98000, "n_cold": 14000, "n_eval_users": 2000},
        "cf": {"popularity": _rank(None), "bias_baseline": _rank(0.91),
               "cf_mf": {**_rank(0.87), "best_epoch": 7, "dim": 32}},
        "content": {"two_tower": {**_rank(0.95), "best_epoch": 4,
                                  "cold": {**_rank(1.01), "n_eval_users": 800}},
                    "cold_baselines": {"user_bias": {"rmse": 1.0567},
                                       "random": {"precision_at_10": 0.0211, "recall_at_10": 0.0322,
                                                  "ndcg_at_10": 0.0277, "n_eval_users": 800}}},
        "clustering": {"k": 3, "silhouette": 0.0712, "names": ["Horror & Thriller fans", "Drama & War fans", "Mainstream"],
                       "sizes": [1500, 2000, 2510]},
        "anomaly": {"epsilon": -12.5, "epsilon_capped": False, "val_f1": 0.9, "test_precision": 0.95,
                    "test_recall": 0.88, "test_f1": 0.7317, "n_bots_test": 150, "n_real_flagged": 5,
                    "n_real_flagged_all": 30,
                    "per_bot_type_recall": {"push": 1.0, "random_spam": 0.0, "camouflage_popular": 0.25},
                    "ablation_plus_item_dev": {"epsilon": -20.5, "val_f1": 0.72, "test_precision": 0.91,
                                               "test_recall": 0.6123, "test_f1": 0.7258,
                                               "per_bot_type_recall": {"push": 1.0, "random_spam": 0.0,
                                                                       "camouflage_popular": 0.0, "nuke": 0.5}}},
        "bandit": {"rounds": 3000, "n_seeds": 5,
                   "policies": {"random": {"mean_reward": 0.05, "cum_regret": 120.0, "cum_regret_std": 4.0},
                                "epsilon_greedy": {"mean_reward": 0.07, "cum_regret": 60.0, "cum_regret_std": 5.0},
                                "linucb": {"mean_reward": 0.08, "cum_regret": 30.0, "cum_regret_std": 3.0}},
                   "oracle_mean_reward": 0.09,
                   "arm_mean_expected_reward": {"cf": 0.06, "content": 0.05, "cluster_popular": 0.075, "trending": 0.04},
                   "frac_users_no_signal": 0.619,
                   "params": {"epsilon": 0.05, "alpha": 0.03, "ridge": 1.0, "tuned_on_users": 1000},
                   "tuning": {"alpha": {"1.0": 58.1, "0.3": 50.2, "0.03": 41.75},
                              "epsilon": {"0.2": 49.0, "0.05": 44.5}}},
        "dqn": None,
    }


DQN = {"dqn": 0.6123, "greedy_cf": 0.58, "random": 0.41, "diversity_heuristic": 0.6, "n_train_users": 190,
       "n_eval_users": 191, "n_users_total": 1000, "n_users_with_signal": 381, "episodes": 3000, "session_len": 5}


def _render(out=None, **kw):
    return render_results_markdown(assemble_metrics(_full_outputs() if out is None else out, _cfg(), None, **kw))


# --------------------------------------------------------------------------- assembly
def test_assemble_merges_rating_models_and_sections():
    m = assemble_metrics(_full_outputs(), _cfg(), {"data": 12.3, "cf": 80.0, "total": 92.3}, generated_at="2026-01-01T00:00:00+00:00")
    assert set(m["rating_models"]) == {"popularity", "bias_baseline", "cf_mf", "two_tower", "cold_baselines"}
    assert m["clustering"]["k"] == 3 and m["bandit"]["rounds"] == 3000
    assert m["dqn"] is None
    assert m["dataset"] == "MovieLens-1M"
    assert m["split"] == ("per-user time-aware 80/10/10 train/val/test, 6% of movies with >= 20 ratings held out "
                          "as cold items, seed 42")
    assert m["fast"] is False
    assert m["train_seconds"] == {"anomaly": None, "data": 12.3, "cf": 80.0, "content": None,
                                  "clustering": None, "bandit": None, "dqn": None}
    json.dumps(m)  # serialisable


def test_assemble_skipped_steps_are_null():
    m = assemble_metrics({"data": {"n_users": 10}, "cf": None}, _cfg(fast=True), None)
    assert m["rating_models"] == {}
    for key in ("clustering", "anomaly", "bandit", "dqn"):
        assert m[key] is None
    assert m["data"] == {"n_users": 10}
    assert m["fast"] is True
    assert all(v is None for v in m["train_seconds"].values())


def test_assemble_cleans_numpy_rounds_and_drops_nan():
    out = {"clustering": {"k": np.int64(4), "silhouette": np.float32(0.123456789),
                          "sizes": np.array([1, 2, 3]), "flag": np.bool_(True), "bad": float("nan")}}
    m = assemble_metrics(out, _cfg(), {})
    c = m["clustering"]
    assert c["k"] == 4 and type(c["k"]) is int
    assert c["silhouette"] == 0.1235
    assert c["sizes"] == [1, 2, 3]
    assert c["flag"] is True
    assert c["bad"] is None
    json.dumps(m)


def test_assemble_cleans_pandas_missing_and_rejects_unknown_types():
    pd = pytest.importorskip("pandas")
    m = assemble_metrics({"data": {"a": pd.NA, "b": pd.NaT, "p": Path("x") / "y"}}, _cfg(), {})
    assert m["data"] == {"a": None, "b": None, "p": "x/y"}
    with pytest.raises(TypeError, match="object"):
        assemble_metrics({"data": {"bad": object()}}, _cfg(), {})


# --------------------------------------------------------------------------- rendering
def test_render_full_metrics_has_4_decimals_and_sections():
    text = _render(generated_at="2026-01-01T00:00:00+00:00")
    assert "| Matrix factorization (CF) | 0.8700 | 0.1234 | 0.0500 | 0.1500 |" in text
    assert "| Popularity | n/a | 0.1234" in text
    assert "cold-start" in text and "1.0100" in text
    assert "Horror & Thriller fans (1,500)" in text
    assert "| LinUCB | 0.0800 | 30.0000 ± 3.0000 |" in text
    assert "DQN stretch goal:** not shipped" in text
    assert "2026-01-01T00:00:00+00:00" in text
    assert "reports/figures/cf_training.png" in text
    assert "reports/figures/elbow.png" in text and "reports/figures/silhouette.png" in text
    assert "**Warm movies**" in text.split("| Model | RMSE")[0]  # warm table is captioned
    assert "ranking-evaluation sample (RMSE uses all held-out rows)" in text
    assert "per-user time-aware 80/10/10 train/val/test, 6% of movies with >= 20 ratings held out" in text
    assert "FAST MODE" not in text


def test_render_headline_results_come_first():
    text = _render()
    head = text.split("**Data:**")[0]
    assert head.startswith("**Headline results**")
    assert "matrix factorization test RMSE 0.8700 vs 0.9100 for the bias baseline" in head
    assert "NDCG@10 0.1500 vs 0.0277 for random ranking." in head
    assert "test F1 0.7317 on synthetic bots (precision 0.9500, recall 0.8800; misses 2 of 3 bot types)" in head
    assert "LinUCB's cumulative regret is 75% below random (30.0000 vs 120.0000" in head
    assert "does not beat always serving the best single strategy" not in head  # 0.08 > best arm 0.075


def test_render_ranked_user_count_when_present():
    assert "Warm ranking metrics average" not in _render()
    out = _full_outputs()
    out["cf"]["n_ranked_users"] = 1904
    assert "average over the 1,904 of them with at least one relevant held-out movie" in _render(out)


def test_render_cold_start_table_with_baselines():
    text = _render()
    assert "| Two-tower content model | 1.0100 | 0.1234 | 0.0500 | 0.1500 |" in text
    assert "| CF fallback: user bias only (mu + b_u, item bias unknown) | 1.0567 | n/a | n/a | n/a |" in text
    assert "| Random ranking (mean over seeds) | n/a | 0.0211 | 0.0322 | 0.0277 |" in text
    main_table = text.split("**Cold-start movies**")[0].split("**Data:**")[1]
    assert "1.0567" not in main_table and "0.0211" not in main_table  # baselines never in the warm table
    assert "cold_baselines" not in text and "Genre popularity" not in text


def test_render_cold_genre_popularity_baseline():
    out = _full_outputs()
    out["content"]["cold_baselines"]["genre_popularity"] = {"precision_at_10": 0.09, "recall_at_10": 0.12,
                                                            "ndcg_at_10": 0.1, "n_eval_users": 800}
    text = _render(out)
    assert "| Genre popularity (non-personalized) | n/a | 0.0900 | 0.1200 | 0.1000 |" in text
    assert "and 0.1000 for a non-personalized genre-popularity ranking." in text
    assert "So 59% of the two-tower's NDCG@10 gain over random comes from these priors" in text
    out["content"]["cold_baselines"]["genre_popularity"]["ndcg_at_10"] = 0.2
    assert "The two-tower does not beat the non-personalized genre-popularity ranking" in _render(out)


def test_render_anomaly_ablation_and_per_type_recall():
    text = _render()
    assert "| Metric | 5 features (deployed) | + item-deviation feature (ablation) |" in text
    assert "| Test recall | 0.8800 | 0.6123 |" in text
    assert "| Test F1 | 0.7317 | 0.7258 |" in text
    assert "| Recall on `camouflage_popular` bots | 0.2500 | 0.0000 |" in text
    assert "| Recall on `nuke` bots | n/a | 0.5000 |" in text  # union of bot types
    assert "misses (test recall below 0.5): `random_spam`, `camouflage_popular`." in text
    assert "did not improve test F1 (0.7258 vs 0.7317)" in text and "mimic real users" in text
    assert "ablation_plus_item_dev" not in text
    assert "Taste profiles:** K-means on MF user vectors, k = 3 (silhouette 0.0712)" in text


def test_render_ablation_that_helps_is_reported_as_such():
    out = _full_outputs()
    out["anomaly"]["ablation_plus_item_dev"]["test_f1"] = 0.8
    text = _render(out)
    assert "raised test F1 from 0.7317 to 0.8000" in text and "did not improve" not in text


def test_render_no_missed_bots_drops_causal_claim():
    out = _full_outputs()
    out["anomaly"]["per_bot_type_recall"] = {"push": 1.0, "random_spam": 0.9, "camouflage_popular": 0.75}
    text = _render(out)
    assert "It did not improve test F1 (0.7258 vs 0.7317)." in text
    assert "mimic real users" not in text and "misses (test recall below 0.5)" not in text


def test_render_epsilons_live_in_details():
    text = _render()
    details = text.split("<details>")[1].split("</details>")[0]
    assert "log-density epsilon -12.5000 (chosen by validation F1); ablation epsilon -20.5000." in details
    assert "-12.5000" not in text.split("<details>")[0]
    out = _full_outputs()
    out["anomaly"]["epsilon_capped"] = True
    out["anomaly"]["ablation_plus_item_dev_capped"] = True
    text = _render(out)
    assert "(the validation-F1 choice, capped so that at most 3% of real users are flagged)" in text
    assert "ablation epsilon -20.5000, capped at the 3% flag limit." in text


def test_render_without_optional_keys_is_graceful():
    """None or missing cold baselines / ablation render n/a, never a fake number."""
    for value in (None, "absent"):
        out = _full_outputs()
        if value == "absent":
            del out["content"]["cold_baselines"], out["anomaly"]["ablation_plus_item_dev"]
        else:
            out["content"]["cold_baselines"] = {"user_bias": None, "random": None}
            out["anomaly"]["ablation_plus_item_dev"] = None
        text = _render(out)
        assert "| CF fallback: user bias only (mu + b_u, item bias unknown) | n/a | n/a | n/a | n/a |" in text
        assert "| Random ranking (mean over seeds) | n/a | n/a | n/a | n/a |" in text
        assert "| Test F1 | 0.7317 | n/a |" in text
        assert "Feature ablation: n/a" in text


def test_render_bandit_tuning_noise_and_no_signal():
    text = _render()
    assert ("tuned on 1,000 simulated users disjoint from the reported ones (lowest mean cumulative regret): "
            "LinUCB alpha = 0.03, epsilon-greedy epsilon = 0.05.") in text
    details = text.split("<details>")[1]
    assert "LinUCB alpha (mean cumulative regret on the tuning users): 1: 58.1000, 0.3: 50.2000, 0.03: 41.7500." in details
    assert "epsilon-greedy epsilon (mean cumulative regret on the tuning users): 0.2: 49.0000, 0.05: 44.5000." in details
    assert "per arm: CF 0.0600, content 0.0500, taste-profile favourites 0.0750, trending 0.0400." in details
    assert "61.9% of simulated users have no click signal" in text
    assert "within seed noise" not in text  # 30 vs 60 with small stds is a real difference
    out = _full_outputs()
    out["bandit"]["policies"]["epsilon_greedy"]["cum_regret"] = 33.0
    assert "LinUCB and epsilon-greedy differ by less than two standard errors over 5 seeds" in _render(out)


def test_render_bandit_best_single_arm_reference():
    out = _full_outputs()
    out["bandit"]["arm_mean_expected_reward"]["cluster_popular"] = 0.085  # above both learners' mean reward
    text = _render(out)
    assert "always serving the best single strategy (taste-profile favourites) has mean expected reward 0.0850, " \
           "above both learners' mean reward" in text
    assert "LinUCB's cumulative regret is 75% below random (30.0000 vs 120.0000, simulated clicks), but it does " \
           "not beat always serving the best single strategy." in text
    out["bandit"]["best_fixed_arm"] = {"arm": "cluster_popular", "mean_reward": 0.085, "cum_regret": 25.0,
                                       "cum_regret_std": 1.0}
    text = _render(out)
    assert "| Always the best single arm (taste-profile favourites; reference, not a learner) | 0.0850 | " \
           "25.0000 ± 1.0000 |" in text
    assert "Neither learner beats always serving the best single strategy (taste-profile favourites)" in text
    out["bandit"]["best_fixed_arm"]["cum_regret"] = 45.0
    text = _render(out)
    assert "- LinUCB beats always serving the best single strategy" in text
    assert "does not beat always serving" not in text


def test_render_bandit_untuned_params():
    out = _full_outputs()
    out["bandit"]["params"] = {"epsilon": 0.1, "alpha": 1.0, "ridge": 1.0}
    del out["bandit"]["tuning"]
    text = _render(out)
    assert "Hyperparameters (config defaults, not tuned): LinUCB alpha = 1, epsilon-greedy epsilon = 0.1." in text
    assert "Tuning grid" not in text


def test_render_dqn_table_and_figure():
    out = _full_outputs()
    out["dqn"] = dict(DQN)
    text = _render(out)
    assert "not shipped" not in text
    assert "order a 5-slot session from 8 CF candidates" in text
    assert "| DQN (greedy policy) | 0.6123 |" in text
    assert "| Diversity heuristic | 0.6000 |" in text
    assert "| CF order | 0.5800 |" in text
    assert "| Random order (mean of 20 orders) | 0.4100 |" in text
    assert "191 held-out users" in text and "190 other users for 3,000 episodes" in text
    assert "Only 381 of 1,000 simulated users (38%) have any click signal" in text
    assert "does not beat" not in text.split("**DQN stretch goal**")[1]
    assert "reports/figures/dqn_training.png" in text
    assert "dqn_training.png" not in _render()
    out["dqn"]["dqn"] = 0.59
    assert "the DQN beats CF order and random order but does not beat the diversity heuristic." in _render(out)


def test_render_dqn_partial_dict_shows_na():
    out = _full_outputs()
    out["dqn"] = {"dqn": 0.5}
    text = _render(out)
    assert "| DQN (greedy policy) | 0.5000 |" in text and "| CF order | n/a |" in text
    assert "Oracle (best" not in text
    out["dqn"]["oracle"] = 0.7
    assert "| Oracle (best 5-item sequence per user, upper bound) | 0.7000 |" in _render(out)


def test_render_random_seed_count_when_present():
    out = _full_outputs()
    out["content"]["cold_baselines"]["random"]["n_seeds"] = 10
    assert "| Random ranking (mean of 10 seeds) | n/a | 0.0211 | 0.0322 | 0.0277 |" in _render(out)


def test_render_cold_note_falls_back_to_random_n_eval_users():
    out = _full_outputs()
    del out["content"]["two_tower"]
    text = _render(out)
    assert "800 sampled users" in text
    cold_table = text.split("**Cold-start movies**")[1]
    assert "| Two-tower content model | n/a | n/a | n/a | n/a |" in cold_table


def test_render_training_time_is_labelled_as_sum():
    text = render_results_markdown(assemble_metrics(_full_outputs(), _cfg(), {"cf": 80.0, "data": 12.3, "total": 99}))
    assert "Training time: 92s, the sum of the step times (data 12s, cf 80s)." in text


def test_render_nulls_as_na_and_not_shipped():
    m = assemble_metrics({}, _cfg(), None, generated_at="x")
    text = render_results_markdown(m)
    assert "n/a" in text and "not shipped" in text
    assert "Headline results" not in text and "<details>" not in text
    assert not FOUR_DECIMALS.search(text)  # no numbers invented for missing models
    assert "reports/figures" not in text   # no links to figures that were never drawn


def test_render_fast_banner():
    text = render_results_markdown(assemble_metrics({}, _cfg(fast=True), None))
    assert "FAST MODE" in text


def _ranking_outputs():
    """Full outputs plus the validation-chosen ranking rows and the new-user simulation (cf eval_step)."""
    out = _full_outputs()
    out["cf"]["cf_mf"]["ndcg_at_10"] = 0.0287
    out["cf"]["popularity"]["ndcg_at_10"] = 0.0532
    out["cf"]["cf_mf_pop"] = {"precision_at_10": 0.06, "recall_at_10": 0.04, "ndcg_at_10": 0.0609, "beta": 1.0}
    out["cf"]["item_knn"] = {"precision_at_10": 0.07, "recall_at_10": 0.05, "ndcg_at_10": 0.0681, "k": 200,
                             "gamma": 0.25}
    out["cf"]["new_users"] = {"n_users": 1000, "n_seed_ratings": 5, "strategies": {
        "popularity": {"precision_at_10": 0.3, "recall_at_10": 0.1, "ndcg_at_10": 0.351},
        "mf_foldin": {"precision_at_10": 0.2, "recall_at_10": 0.08, "ndcg_at_10": 0.222},
        "mf_foldin_pop": {"precision_at_10": 0.31, "recall_at_10": 0.11, "ndcg_at_10": 0.36},
        "item_knn": {"precision_at_10": 0.32, "recall_at_10": 0.12, "ndcg_at_10": 0.37}}}
    return out


def test_render_ranking_rows_with_validation_chosen_values():
    text = _render(_ranking_outputs())
    table = text.split("**Warm movies**")[1].split("\n\n- ")[0]
    rows = [r.split(" | ")[0].lstrip("| ") for r in table.splitlines() if r.startswith("| ") and "---" not in r][1:]
    assert rows == ["Popularity", "Bias baseline (mu + b_u + b_i)",
                    "Matrix factorization (ranked by predicted rating)",
                    "MF + popularity prior (beta = 1, chosen on validation)",
                    "Item-kNN (k = 200, chosen on validation)", "Two-tower content model"]
    assert "| MF + popularity prior (beta = 1, chosen on validation) | n/a | 0.0600 | 0.0400 | 0.0609 |" in text
    assert "| Item-kNN (k = 200, chosen on validation) | n/a | 0.0700 | 0.0500 | 0.0681 |" in text
    assert "gamma = 0.25" in text and "chosen by NDCG@10 on the validation split" in text
    assert "Matrix factorization (CF)" not in text


def test_render_new_user_table_and_ranking_headlines():
    text = _render(_ranking_outputs())
    head = text.split("**Data:**")[0]
    assert ("**Top-10 ranking** (warm movies): NDCG@10 item-kNN 0.0681, MF + popularity prior 0.0609, "
            "popularity 0.0532, MF ranked by its predicted rating 0.0287.") in head
    assert "**New users** (simulated, 5 known ratings): item-kNN NDCG@10 0.3700 vs 0.3510 for popularity." in head
    section = text.split("**New users (simulated, 5 known ratings)**")[1].split("**Cold-start movies**")[0]
    assert "| Strategy | P@10 | R@10 | NDCG@10 |" in section
    assert "| Popularity | 0.3000 | 0.1000 | 0.3510 |" in section
    assert "| MF fold-in (ranked by predicted rating) | 0.2000 | 0.0800 | 0.2220 |" in section
    assert "| MF fold-in + popularity prior | 0.3100 | 0.1100 | 0.3600 |" in section
    assert "| Item-kNN on the liked seed movies | 0.3200 | 0.1200 | 0.3700 |" in section
    assert "1,000 sampled users" in section


def test_render_non_model_keys_never_in_main_table():
    """cold_baselines / new_users / n_ranked_users live in rating_models but are not models."""
    out = _ranking_outputs()
    out["cf"]["n_ranked_users"] = 1904
    text = _render(out)
    main = text.split("**Warm movies**")[1].split("**New users (simulated")[0]
    for key in ("new_users", "cold_baselines", "n_ranked_users", "strategies"):
        assert key not in text
    assert "0.3510" not in main and "1.0567" not in main and "0.0211" not in main


def test_render_without_ranking_keys_keeps_old_output():
    """Old metrics.json (no cf_mf_pop / item_knn / new_users): same table, no new sections or headlines."""
    text = _render()
    assert "| Matrix factorization (CF) | 0.8700 |" in text
    assert "ranked by predicted rating" not in text and "Item-kNN" not in text
    assert "New users" not in text and "Top-10 ranking" not in text and "chosen on validation" not in text
    out = _ranking_outputs()
    out["cf"]["new_users"] = {"n_users": 10, "n_seed_ratings": 5, "strategies": {}}
    del out["cf"]["cf_mf_pop"]
    out["cf"]["item_knn"] = {"ndcg_at_10": None}           # partial dicts render n/a, never crash
    text = _render(out)
    assert "**New users (simulated" not in text and "**New users**" not in text
    assert "| Item-kNN (k = n/a, chosen on validation) | n/a | n/a | n/a | n/a |" in text
    assert "MF + popularity prior" not in text and "Top-10 ranking" not in text


def test_fmt():
    assert fmt(None) == "n/a"
    assert fmt(0.5) == "0.5000"
    assert fmt(np.float32(0.12345678)) == "0.1235"


# --------------------------------------------------------------------------- README update
def _readme(tmp_path, body="old block"):
    p = tmp_path / "README.md"
    p.write_text(f"# Title\n\nMentions `{MARKER_START}` inline.\n\n{MARKER_START}\n{body}\n{MARKER_END}\n\n## After\ntext\n",
                 encoding="utf-8")
    return p


def test_update_readme_replaces_block_and_is_idempotent(tmp_path):
    p = _readme(tmp_path)
    update_readme(p, "NEW 1.2345")
    once = p.read_text(encoding="utf-8")
    update_readme(p, "NEW 1.2345")
    assert p.read_text(encoding="utf-8") == once
    head, block, tail = split_readme(once)
    assert block.strip() == "NEW 1.2345"
    assert "old block" not in once
    assert head.startswith("# Title") and "inline" in head
    assert tail.startswith(MARKER_END) and "## After\ntext" in tail


def test_update_readme_missing_markers_raises(tmp_path):
    p = tmp_path / "README.md"
    p.write_text(f"# no markers here, only inline `{MARKER_START}`\n", encoding="utf-8")
    with pytest.raises(ValueError):
        update_readme(p, "x")
    p.write_text(f"{MARKER_END}\nbefore start\n{MARKER_START}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        update_readme(p, "x")


def test_repo_readme_has_no_hand_typed_metrics():
    """Number-provenance rule: 4-decimal numbers may only appear inside the markers."""
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    head, _, tail = split_readme(text)
    outside = head + tail
    assert not FOUR_DECIMALS.findall(outside), FOUR_DECIMALS.findall(outside)


# --------------------------------------------------------------------------- step runner
def test_step_runner_records_outputs_and_fails_on_unknown_steps(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    import train_all  # a broken import must fail, not skip
    monkeypatch.setattr(train_all, "set_seed", lambda seed: None)  # avoid loading torch

    fake = types.ModuleType("cinematch.fake_step")
    fake.eval_step = lambda cfg, ctx: {"k": 1}
    no_fn = types.ModuleType("cinematch.fake_nofn")
    monkeypatch.setitem(sys.modules, "cinematch.fake_step", fake)
    monkeypatch.setitem(sys.modules, "cinematch.fake_nofn", no_fn)

    ctx = train_all.run(["fake_step"], "eval_step", _cfg())
    assert ctx["outputs"] == {"fake_step": {"k": 1}}
    assert set(ctx["timings"]) == {"fake_step"}
    m = assemble_metrics({"clustering": ctx["outputs"]["fake_step"]}, _cfg(), ctx["timings"])
    assert m["clustering"] == {"k": 1}
    with pytest.raises(ModuleNotFoundError):  # a typo in the step list is an error, not a silent skip
        train_all.run(["does_not_exist_xyz"], "eval_step", _cfg())
    with pytest.raises(AttributeError):
        train_all.run(["fake_nofn"], "eval_step", _cfg())


# --------------------------------------------------------------------------- scripts end-to-end
def _patch_scripts(monkeypatch, tmp_path, order):
    """Import train_all/evaluate with every repo path redirected into tmp_path."""
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    monkeypatch.delenv("CINEMATCH_FAST", raising=False)
    import evaluate
    import train_all

    fake = types.ModuleType("cinematch.fake_e2e")
    fake.train_step = lambda cfg, ctx: None
    fake.eval_step = lambda cfg, ctx: {"n_users": 10}
    monkeypatch.setitem(sys.modules, "cinematch.fake_e2e", fake)
    monkeypatch.setattr(train_all, "set_seed", lambda seed: None)  # avoid loading torch
    monkeypatch.setattr(train_all, "ensure_dirs", lambda: None)
    monkeypatch.setattr(train_all, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(train_all, "TRAIN_ORDER", order)
    monkeypatch.setattr(evaluate, "ensure_dirs", lambda: None)
    monkeypatch.setattr(evaluate, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(evaluate, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(evaluate, "README_PATH", _readme(tmp_path))
    monkeypatch.setattr(evaluate, "EVAL_ORDER", order)
    return train_all, evaluate


def test_train_all_writes_timings(monkeypatch, tmp_path):
    train_all, _ = _patch_scripts(monkeypatch, tmp_path, ["fake_e2e"])
    train_all.main(["--fast"])
    timings = json.loads((tmp_path / "train_timings.json").read_text(encoding="utf-8"))
    assert set(timings) == {"fake_e2e", "total", "fast"}
    assert timings["fast"] is True


def test_evaluate_writes_null_metrics_and_readme(monkeypatch, tmp_path):
    """No steps run (as if every eval_step found its artifacts missing): everything is null."""
    _, evaluate = _patch_scripts(monkeypatch, tmp_path, [])
    evaluate.main(["--fast"])
    m = json.loads((tmp_path / "reports" / "metrics.json").read_text(encoding="utf-8"))
    assert set(m) == {"generated_at", "dataset", "split", "fast", "train_seconds", "data",
                      "rating_models", "clustering", "anomaly", "bandit", "dqn"}
    assert m["fast"] is True and m["rating_models"] == {}
    assert all(m[k] is None for k in ("data", "clustering", "anomaly", "bandit", "dqn"))
    assert all(v is None for v in m["train_seconds"].values())
    _, block, _ = split_readme(evaluate.README_PATH.read_text(encoding="utf-8"))
    assert "FAST MODE" in block and "not shipped" in block and m["generated_at"] in block


def test_evaluate_flags_fast_trained_models(monkeypatch, tmp_path):
    train_all, evaluate = _patch_scripts(monkeypatch, tmp_path, ["fake_e2e"])
    train_all.main(["--fast"])
    m = evaluate.main([])  # full evaluate on models trained with --fast
    assert m["fast"] is True
    assert m["fake_e2e"] == {"n_users": 10}
    assert "FAST MODE" in evaluate.README_PATH.read_text(encoding="utf-8")
