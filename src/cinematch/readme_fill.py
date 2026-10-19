"""Assemble ``reports/metrics.json`` and render the README results block.

Pure Python (no numpy/torch import) so it is cheap to import and easy to test.
Every metric is rendered with exactly 4 decimals; nothing in the README outside
the RESULTS markers is allowed to contain such a number (number-provenance rule).
"""
from __future__ import annotations

import math
import numbers
import re
from datetime import datetime, timezone
from pathlib import Path, PurePath
from typing import Any

MARKER_START = "<!-- RESULTS:START -->"
MARKER_END = "<!-- RESULTS:END -->"

# Steps whose wall-clock training time is reported (fixed key set in metrics.json).
TRAIN_STEPS = ["anomaly", "data", "cf", "content", "clustering", "bandit", "dqn"]
# eval_step outputs merged into metrics["rating_models"]; every other step gets its own key.
RATING_MODEL_STEPS = ("cf", "content")
SECTION_KEYS = ("data", "clustering", "anomaly", "bandit", "dqn")

MODEL_LABELS = {
    "popularity": "Popularity",
    "bias_baseline": "Bias baseline (mu + b_u + b_i)",
    "cf_mf": "Matrix factorization (CF)",
    "two_tower": "Two-tower content model",
}
# Order of the rows in the main warm table. Only these keys are ever rendered there:
# other rating_models keys (cold_baselines, new_users, n_ranked_users) are not models.
WARM_TABLE_KEYS = ("popularity", "bias_baseline", "cf_mf", "cf_mf_pop", "item_knn", "two_tower")
# Once a ranking rule other than "predicted rating" is reported, the plain MF row says how it ranks.
MF_RANKED_BY_RATING = "Matrix factorization (ranked by predicted rating)"
NEW_USER_LABELS = {
    "popularity": "Popularity",
    "mf_foldin": "MF fold-in (ranked by predicted rating)",
    "mf_foldin_pop": "MF fold-in + popularity prior",
    "item_knn": "Item-kNN on the liked seed movies",
}
POLICY_LABELS = {
    "random": "Random",
    "epsilon_greedy": "Epsilon-greedy",
    "linucb": "LinUCB",
}
ARM_LABELS = {
    "cf": "CF",
    "content": "content",
    "cluster_popular": "taste-profile favourites",
    "trending": "trending",
}
RANK_METRICS = ("precision_at_10", "recall_at_10", "ndcg_at_10")


# --------------------------------------------------------------------------- assembly
def _clean(obj: Any) -> Any:
    """Make step outputs JSON-safe: numpy scalars/arrays -> Python, floats rounded to
    4 decimals, NaN/inf/NA -> None (never a fake number). Unknown types raise
    ``TypeError`` naming the type, so the offending eval_step is easy to find."""
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if type(obj).__name__ in ("NAType", "NaTType"):
        return None  # pandas missing values (pd.NA, pd.NaT)
    if hasattr(obj, "tolist") and not isinstance(obj, numbers.Number):
        return _clean(obj.tolist())  # numpy arrays and scalars (incl. np.bool_)
    if isinstance(obj, numbers.Integral):
        return int(obj)
    if isinstance(obj, numbers.Real):
        value = float(obj)
        return round(value, 4) if math.isfinite(value) else None
    if isinstance(obj, PurePath):
        return obj.as_posix()
    if hasattr(obj, "isoformat"):
        return obj.isoformat()  # datetime / pandas Timestamp
    raise TypeError(f"metrics value of type {type(obj).__name__} is not JSON-serialisable: {obj!r}")


def assemble_metrics(
    step_outputs: dict[str, Any],
    cfg: Any,
    train_seconds: dict[str, Any] | None = None,
    generated_at: str | None = None,
) -> dict:
    """Build the metrics.json dict from ``{step_name: eval_step output}``.

    Assembly rule: ``cf``/``content`` outputs are merged into ``rating_models``; every
    other step's output is stored under its own name. Steps that were skipped (absent
    from ``step_outputs``) or returned ``None`` leave ``null``.
    """
    seconds = {name: None for name in TRAIN_STEPS}
    for name, value in (train_seconds or {}).items():
        if name in seconds:
            seconds[name] = value

    val_frac, test_frac = getattr(cfg, "val_frac", 0.10), getattr(cfg, "test_frac", 0.10)
    split = (
        f"per-user time-aware {round(100 * (1 - val_frac - test_frac))}/{round(100 * val_frac)}/"
        f"{round(100 * test_frac)} train/val/test, {round(100 * getattr(cfg, 'cold_item_frac', 0.06))}% of movies "
        f"with >= {getattr(cfg, 'cold_item_min_ratings', 20)} ratings held out as cold items, "
        f"seed {getattr(cfg, 'seed', 42)}"
    )
    metrics: dict[str, Any] = {
        "generated_at": generated_at or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": "MovieLens-1M",
        "split": split,
        "fast": bool(getattr(cfg, "fast", False)),
        "train_seconds": seconds,
        "data": None,
        "rating_models": {},
        "clustering": None,
        "anomaly": None,
        "bandit": None,
        "dqn": None,
    }
    for name, out in step_outputs.items():
        if out is None:
            continue
        if name in RATING_MODEL_STEPS:
            metrics["rating_models"].update(out)
        else:
            metrics[name] = out
    return _clean(metrics)


# --------------------------------------------------------------------------- rendering
def fmt(value: Any) -> str:
    """A metric with exactly 4 decimals, or ``n/a`` when missing."""
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


def _count(value: Any) -> str:
    return "n/a" if value is None else f"{int(value):,}"


def _model_row(label: str, m: dict | None) -> str:
    m = m or {}
    cells = [fmt(m.get(k)) for k in ("rmse", "precision_at_10", "recall_at_10", "ndcg_at_10")]
    return f"| {label} | " + " | ".join(cells) + " |"


def _n_ranked_users(rm: dict) -> Any:
    """Users that actually enter the warm ranking average (those with >= 1 relevant test item)."""
    return rm.get("n_ranked_users", (rm.get("cf_mf") or {}).get("n_eval_users"))


def _data_sentence(metrics: dict) -> str:
    d = metrics.get("data")
    split = metrics.get("split", "")
    if not d:
        return f"**Data:** {metrics.get('dataset', 'MovieLens-1M')} ({split}); split statistics not available."
    n_ranked = _n_ranked_users(metrics.get("rating_models") or {})
    ranked = (f" Warm ranking metrics average over the {_count(n_ranked)} of them with at least one relevant held-out "
              "movie (users without one are skipped)." if n_ranked is not None else "")
    return (
        f"**Data:** {metrics.get('dataset', 'MovieLens-1M')}, {_count(d.get('n_users'))} users, "
        f"{_count(d.get('n_items'))} movies ({_count(d.get('n_cold_items'))} held out as cold-start items). "
        f"Split: {split}: {_count(d.get('n_train'))} train / {_count(d.get('n_val'))} validation / "
        f"{_count(d.get('n_test'))} test / {_count(d.get('n_cold'))} cold ratings. "
        f"{_count(d.get('n_flagged_users'))} users flagged by anomaly detection are excluded from model training "
        f"and from the ranking-evaluation sample (RMSE uses all held-out rows). Ranking metrics use a fixed seeded sample of {_count(d.get('n_eval_users'))} test users; "
        f"a held-out rating >= 4 counts as relevant.{ranked}"
    )


def _has_ranking_rows(rm: dict) -> bool:
    """True when metrics.json reports the validation-chosen ranking rules (MF + prior, item-kNN)."""
    return bool(rm.get("cf_mf_pop") or rm.get("item_knn"))


def _warm_label(key: str, rm: dict) -> str:
    """Row label of one warm-table model; the ranking rows carry their validation-chosen value."""
    m = rm.get(key) or {}
    if key == "cf_mf_pop":
        return f"MF + popularity prior (beta = {_num(m.get('beta'))}, chosen on validation)"
    if key == "item_knn":
        return f"Item-kNN (k = {_count(m.get('k'))}, chosen on validation)"
    if key == "cf_mf" and _has_ranking_rows(rm):
        return MF_RANKED_BY_RATING
    return MODEL_LABELS[key]


def _rating_section(rm: dict) -> list[str]:
    """Main warm-item table: only the model keys in WARM_TABLE_KEYS (never ``cold_baselines`` / ``new_users``).

    The base four models always get a row (n/a when missing); the ranking rows only when present.
    """
    lines = [
        "**Warm movies** (test split; ranking over warm candidates not in the user's train set):",
        "",
        "| Model | RMSE | P@10 | R@10 | NDCG@10 |",
        "|---|---|---|---|---|",
    ]
    for key in WARM_TABLE_KEYS:
        if key in MODEL_LABELS or rm.get(key):
            lines.append(_model_row(_warm_label(key, rm), rm.get(key)))
    lines.append("")
    notes = []
    mf = rm.get("cf_mf")
    if mf:
        notes.append(f"MF: dim {mf.get('dim', 'n/a')}, best epoch {mf.get('best_epoch', 'n/a')} (early stopping on validation RMSE).")
    notes.append("Popularity has no rating prediction (RMSE n/a).")
    if rm.get("cf_mf_pop"):
        notes.append("MF + popularity prior is the same MF model ranked by predicted rating + beta x log(1 + the "
                     "movie's train rating count); its rating predictions are MF's.")
    knn = rm.get("item_knn")
    if knn:
        notes.append("Item-kNN scores a movie by its cosine similarity to the movies the user rated >= 4 (top-k "
                     f"neighbours per movie) + gamma x log(1 + train rating count), gamma = {_num(knn.get('gamma'))}; "
                     "it predicts no rating (RMSE n/a).")
    if _has_ranking_rows(rm):
        notes.append("beta, k and gamma were chosen by NDCG@10 on the validation split; the test split was not used to choose them.")
    lines.extend(f"- {n}" for n in notes)
    return lines


def _new_user_section(rm: dict) -> list[str]:
    """Simulated new users (the app's rate-a-few-movies scenario), or nothing when not evaluated."""
    nu = rm.get("new_users") or {}
    strategies = nu.get("strategies") or {}
    if not strategies:
        return []
    n_seeds = nu.get("n_seed_ratings", "n/a")
    lines = [
        f"**New users (simulated, {n_seeds} known ratings)** (test split; the app's rate-a-few-movies flow):",
        "",
        "| Strategy | P@10 | R@10 | NDCG@10 |",
        "|---|---|---|---|",
    ]
    for key, m in strategies.items():
        m = m or {}
        cells = " | ".join(fmt(m.get(k)) for k in RANK_METRICS)
        lines.append(f"| {NEW_USER_LABELS.get(key, key)} | {cells} |")
    lines += [
        "",
        f"- {_count(nu.get('n_users'))} sampled users; each one's only known ratings are {n_seeds} random train "
        "ratings of well-known movies (many train ratings), scored exactly as the app scores a new user (MF fold-in, or item-kNN on the "
        "seeds rated >= 4, else >= 3, else popularity). Candidates are well-known warm movies the user has not "
        "rated in train; a test rating >= 4 counts as relevant.",
    ]
    return lines


def _cold_section(rm: dict) -> list[str]:
    """Cold-start table: two-tower vs the user-bias RMSE fallback, a non-personalized genre-popularity
    ranking (when evaluated) and random ranking, all on the same users and cold candidates."""
    cold = (rm.get("two_tower") or {}).get("cold")
    baselines = rm.get("cold_baselines") or {}
    user_bias = baselines.get("user_bias") or {}
    genre_pop = baselines.get("genre_popularity")
    rand = baselines.get("random") or {}
    n_seeds = rand.get("n_seeds")
    rand_label = f"Random ranking (mean of {n_seeds} seeds)" if n_seeds else "Random ranking (mean over seeds)"
    lines = [
        "**Cold-start movies** (never seen in training; MF cannot score them by construction):",
        "",
        "| Cold-start model | RMSE | P@10 | R@10 | NDCG@10 |",
        "|---|---|---|---|---|",
        _model_row("Two-tower content model", cold),
        _model_row("CF fallback: user bias only (mu + b_u, item bias unknown)", {"rmse": user_bias.get("rmse")}),
    ]
    if genre_pop:
        lines.append(_model_row("Genre popularity (non-personalized)", {k: genre_pop.get(k) for k in RANK_METRICS}))
    lines += [_model_row(rand_label, {k: rand.get(k) for k in RANK_METRICS}), ""]
    n_users = (cold or {}).get("n_eval_users", rand.get("n_eval_users"))
    lines.append(
        f"- Ranking over cold candidates only, {_count(n_users)} sampled users with a relevant cold rating. "
        "The CF fallback row is what the bias/MF models predict for an unseen movie (item bias unknown; "
        "RMSE only, it gives every unseen movie the same score, so it cannot rank); "
        "the other rankings use exactly the same users and candidates."
    )
    lines += _cold_prior_note(cold or {}, genre_pop or {}, rand)
    return lines


def _cold_prior_note(cold: dict, genre_pop: dict, rand: dict) -> list[str]:
    """How much of the two-tower's cold-start NDCG gain over random a genre-popularity prior already gets."""
    tt, gp, rnd = (m.get("ndcg_at_10") for m in (cold, genre_pop, rand))
    if tt is None or gp is None:
        return []
    tt, gp = float(tt), float(gp)
    if tt <= gp:
        return [f"- The two-tower does not beat the non-personalized genre-popularity ranking on NDCG@10 "
                f"({fmt(tt)} vs {fmt(gp)}): its cold-start lift over random comes from genre and popularity priors."]
    note = (f"- Genre and popularity priors alone (no personalization) reach NDCG@10 {fmt(gp)} vs the "
            f"two-tower's {fmt(tt)}.")
    if rnd is not None and tt > float(rnd):
        share = 100 * (gp - float(rnd)) / (tt - float(rnd))
        note += (f" So {share:.0f}% of the two-tower's NDCG@10 gain over random comes from these priors and the rest "
                 "from personalization.")
    return [note]


DETECTOR_METRICS = (("val_f1", "Validation F1"), ("test_precision", "Test precision"),
                    ("test_recall", "Test recall"), ("test_f1", "Test F1"))


MISSED_RECALL = 0.5   # a bot type with deployed test recall below this is called "missed"


def _anomaly_section(a: dict | None) -> list[str]:
    """Anomaly line plus the feature-ablation table with per-bot-type recall."""
    if not a:
        return ["**Anomaly detection:** n/a (not run)."]
    abl = a.get("ablation_plus_item_dev") or {}
    lines = [
        f"**Anomaly detection:** Gaussian density over 5 behaviour features; a user is flagged when the log-density "
        f"falls below a threshold epsilon chosen by validation F1. "
        f"Test set: {_count(a.get('n_bots_test'))} synthetic bots + held-out real users. "
        f"Real users flagged and excluded: {_count(a.get('n_real_flagged_all'))}.",
        "",
        "| Metric | 5 features (deployed) | + item-deviation feature (ablation) |",
        "|---|---|---|",
    ]
    for key, label in DETECTOR_METRICS:
        lines.append(f"| {label} | {fmt(a.get(key))} | {fmt(abl.get(key))} |")
    per_dep = a.get("per_bot_type_recall") or {}
    per_abl = abl.get("per_bot_type_recall") or {}
    for t in list(per_dep) + [t for t in per_abl if t not in per_dep]:
        lines.append(f"| Recall on `{t}` bots | {fmt(per_dep.get(t))} | {fmt(per_abl.get(t))} |")
    lines.append("")
    missed = [t for t, r in per_dep.items() if r is not None and float(r) < MISSED_RECALL]
    if missed:
        lines.append("- Bot types the deployed model misses (test recall below 0.5): "
                     + ", ".join(f"`{t}`" for t in missed) + ".")
    if not abl:
        lines.append("- Feature ablation: n/a (not in metrics.json).")
        return lines
    f1_dep, f1_abl = a.get("test_f1"), abl.get("test_f1")
    if f1_dep is None or f1_abl is None:
        verdict = ""
    elif float(f1_abl) > float(f1_dep):
        verdict = f" It raised test F1 from {fmt(f1_dep)} to {fmt(f1_abl)}."
    elif missed:
        verdict = (f" It did not improve test F1 ({fmt(f1_abl)} vs {fmt(f1_dep)}): the missed bots mimic real "
                   "users' rating statistics, so catching them needs item-level / co-rating graph signals.")
    else:
        verdict = f" It did not improve test F1 ({fmt(f1_abl)} vs {fmt(f1_dep)})."
    lines.append(
        "- Ablation: the same protocol with a 6th engineered feature, the mean absolute deviation of a user's "
        f"ratings from each movie's mean, tried after error analysis on validation bots.{verdict} Test bots come "
        "from the same synthetic generator, so this measures detection of these patterns, not real-world abuse."
    )
    return lines


def _anomaly_details(a: dict | None) -> list[str]:
    """Exact epsilon thresholds (log-density units), kept out of the main text."""
    if not a:
        return []
    capped = " (the validation-F1 choice, capped so that at most 3% of real users are flagged)"         if a.get("epsilon_capped") else " (chosen by validation F1)"
    line = f"- Anomaly threshold: log-density epsilon {fmt(a.get('epsilon'))}{capped}"
    abl = a.get("ablation_plus_item_dev") or {}
    if abl.get("epsilon") is not None:
        abl_capped = ", capped at the 3% flag limit" if a.get("ablation_plus_item_dev_capped") else ""
        line += f"; ablation epsilon {fmt(abl['epsilon'])}{abl_capped}"
    return [line + "."]


def _clustering_line(c: dict | None) -> str:
    if not c:
        return "**Taste profiles:** n/a (not run)."
    names = c.get("names") or []
    sizes = c.get("sizes") or []
    profiles = ", ".join(f"{n} ({_count(s)})" for n, s in zip(names, sizes))
    return (
        f"**Taste profiles:** K-means on MF user vectors, k = {c.get('k', 'n/a')} "
        f"(silhouette {fmt(c.get('silhouette'))}): {profiles}."
    )


def _arm_label(name: str) -> str:
    return ARM_LABELS.get(name, name)


def _best_arm(b: dict) -> tuple[str, Any] | None:
    """(arm name, mean expected reward) of the best single arm, or None."""
    arms = {k: v for k, v in (b.get("arm_mean_expected_reward") or {}).items() if v is not None}
    if not arms:
        return None
    name = max(arms, key=lambda k: float(arms[k]))
    return name, arms[name]


def _bandit_section(b: dict | None) -> list[str]:
    if not b:
        return ["**Bandit:** n/a (not run)."]
    lines = [
        f"**Contextual bandit** ({_count(b.get('rounds'))} rounds x {_count(b.get('n_seeds'))} seeds, simulated clicks):",
        "",
        "| Policy | Mean reward | Cumulative regret (± std) |",
        "|---|---|---|",
    ]
    for key, p in (b.get("policies") or {}).items():
        p = p or {}
        lines.append(
            f"| {POLICY_LABELS.get(key, key)} | {fmt(p.get('mean_reward'))} | "
            f"{fmt(p.get('cum_regret'))} ± {fmt(p.get('cum_regret_std'))} |"
        )
    fixed = b.get("best_fixed_arm")
    if fixed:
        lines.append(
            f"| Always the best single arm ({_arm_label(fixed.get('arm', 'n/a'))}; reference, not a learner) | "
            f"{fmt(fixed.get('mean_reward'))} | {fmt(fixed.get('cum_regret'))} ± {fmt(fixed.get('cum_regret_std'))} |"
        )
    lines.append("")
    lines += _bandit_verdict(b)
    params = b.get("params") or {}
    if params:
        values = f"LinUCB alpha = {_num(params.get('alpha'))}, epsilon-greedy epsilon = {_num(params.get('epsilon'))}"
        n_tune = params.get("tuned_on_users")
        if n_tune:
            lines.append(f"- Hyperparameters tuned on {_count(n_tune)} simulated users disjoint from the reported ones "
                         f"(lowest mean cumulative regret): {values}.")
        else:
            lines.append(f"- Hyperparameters (config defaults, not tuned): {values}.")
    frac = b.get("frac_users_no_signal")
    if frac is not None:
        lines.append(
            f"- {100 * float(frac):.1f}% of simulated users have no click signal (none of their arms' lists contains a "
            "movie they rated in the test split, so every arm has the same expected reward); only the remaining users "
            "can separate the policies."
        )
    return lines


def _bandit_verdict(b: dict) -> list[str]:
    """Plain-language reading of the bandit table: vs the best single arm, and LinUCB vs epsilon-greedy."""
    pol = b.get("policies") or {}
    learners = {POLICY_LABELS[k]: pol.get(k) or {} for k in ("linucb", "epsilon_greedy")}
    rounds = _count(b.get("rounds"))
    out = []
    fixed = b.get("best_fixed_arm") or {}
    best = _best_arm(b)
    if fixed.get("cum_regret") is not None:
        arm = _arm_label(fixed.get("arm", "n/a"))
        winners = [n for n, p in learners.items()
                   if p.get("cum_regret") is not None and float(p["cum_regret"]) < float(fixed["cum_regret"])]
        if not winners:
            out.append(f"- Neither learner beats always serving the best single strategy ({arm}) within {rounds} "
                       "rounds, so the user context does not pay off yet in this simulation.")
        else:
            out.append(f"- {' and '.join(winners)} beat{'s' if len(winners) == 1 else ''} always serving the best "
                       f"single strategy ({arm}) on cumulative regret.")
    elif best is not None:
        rewards = [p.get("mean_reward") for p in learners.values() if p.get("mean_reward") is not None]
        note = (f"- Reference: always serving the best single strategy ({_arm_label(best[0])}) has mean expected "
                f"reward {fmt(best[1])}")
        if rewards and float(best[1]) > max(float(r) for r in rewards):
            note += (f", above both learners' mean reward, so within {rounds} rounds the user context does not pay "
                     "off yet in this simulation")
        out.append(note + ".")
    lin, eps = learners.values()
    n_seeds = b.get("n_seeds")
    if n_seeds and all(p.get(k) is not None for p in (lin, eps) for k in ("cum_regret", "cum_regret_std")):
        se = math.sqrt((float(lin["cum_regret_std"]) ** 2 + float(eps["cum_regret_std"]) ** 2) / int(n_seeds))
        if abs(float(lin["cum_regret"]) - float(eps["cum_regret"])) < 2 * se:
            out.append(f"- LinUCB and epsilon-greedy differ by less than two standard errors over {_count(n_seeds)} "
                       "seeds, i.e. within seed noise.")
    return out


def _bandit_details(b: dict | None) -> list[str]:
    """Oracle / per-arm rewards and the tuning grids, kept out of the main text."""
    if not b:
        return []
    arms = b.get("arm_mean_expected_reward") or {}
    arm_text = ", ".join(f"{_arm_label(name)} {fmt(v)}" for name, v in arms.items())
    lines = [f"- Bandit: oracle mean reward {fmt(b.get('oracle_mean_reward'))}; mean expected reward per arm: "
             f"{arm_text or 'n/a'}."]
    tuning = b.get("tuning") or {}
    for key, label in (("alpha", "LinUCB alpha"), ("epsilon", "epsilon-greedy epsilon")):
        grid = tuning.get(key) or {}
        if grid:
            cells = ", ".join(f"{_num(v)}: {fmt(r)}" for v, r in grid.items())
            lines.append(f"- Tuning grid for {label} (mean cumulative regret on the tuning users): {cells}.")
    return lines


def _num(value: Any) -> str:
    """A hyperparameter value in its shortest form (0.03, 1), or ``n/a``."""
    return "n/a" if value is None else f"{float(value):g}"


DQN_LABELS = (("dqn", "DQN (greedy policy)"), ("diversity_heuristic", "Diversity heuristic"),
              ("greedy_cf", "CF order"), ("random", "Random order (mean of 20 orders)"))
DQN_RIVALS = (("diversity_heuristic", "the diversity heuristic"), ("greedy_cf", "CF order"),
              ("random", "random order"))


def _dqn_section(d: dict | None) -> list[str]:
    """DQN session-ordering results, or "not shipped"."""
    if not d:
        return ["**DQN stretch goal:** not shipped."]
    lines = [
        f"**DQN stretch goal** (order a {d.get('session_len', 'n/a')}-slot session from 8 CF candidates; "
        "simulated position decay and genre fatigue make the order matter):",
        "",
        "| Ordering policy | Mean expected session reward |",
        "|---|---|",
    ]
    lines += [f"| {label} | {fmt(d.get(key))} |" for key, label in DQN_LABELS]
    if d.get("oracle") is not None:
        lines.append(f"| Oracle (best 5-item sequence per user, upper bound) | {fmt(d['oracle'])} |")
    signal = ""
    if d.get("n_users_with_signal") is not None and d.get("n_users_total"):
        share = 100 * d["n_users_with_signal"] / d["n_users_total"]
        signal = (f" Only {_count(d['n_users_with_signal'])} of {_count(d['n_users_total'])} simulated users "
                  f"({share:.0f}%) have any click signal among the candidates.")
    lines += [
        "",
        f"- Expected (analytic) clicks per session on {_count(d.get('n_eval_users'))} held-out users with click "
        f"signal; the DQN was trained on {_count(d.get('n_train_users'))} other users for "
        f"{_count(d.get('episodes'))} episodes.{signal}",
    ]
    if d.get("dqn") is not None:
        rivals = [(name, d[key]) for key, name in DQN_RIVALS if d.get(key) is not None]
        beats = [name for name, v in rivals if d["dqn"] > v]
        loses = [name for name, v in rivals if d["dqn"] <= v]
        if loses:
            lead = f"beats {' and '.join(beats)} but " if beats else ""
            lines.append(f"- On held-out users the DQN {lead}does not beat {' and '.join(loses)}.")
    return lines


def _figures(metrics: dict) -> list[tuple[str, str]]:
    """(alt text, file name) of the key figures, only when the step that draws them produced metrics."""
    rm = metrics.get("rating_models") or {}
    figs = []
    if rm.get("cf_mf"):
        figs.append(("MF training curve", "cf_training.png"))
    if rm.get("two_tower"):
        figs.append(("Two-tower training curve", "two_tower_training.png"))
    if metrics.get("clustering"):
        figs.append(("Taste profiles (PCA of MF user vectors)", "clusters_pca.png"))
        figs.append(("Silhouette vs k", "silhouette.png"))
        figs.append(("Elbow (inertia vs k)", "elbow.png"))
    if metrics.get("anomaly"):
        figs.append(("Anomaly scores: real users vs bots", "anomaly_scores.png"))
    if metrics.get("bandit"):
        figs.append(("Bandit cumulative regret", "bandit_regret.png"))
        figs.append(("Bandit cumulative reward", "bandit_cumulative_reward.png"))
    if metrics.get("dqn"):
        figs.append(("DQN training curve", "dqn_training.png"))
    return figs


def _figure_grid(figs: list[tuple[str, str]]) -> list[str]:
    """Figures two per row as HTML <img> tags (GitHub renders them)."""
    return [f'<img src="reports/figures/{name}" alt="{alt}" width="49%">' for alt, name in figs]


def _context_pays_off(b: dict) -> bool | None:
    """Does the better learner beat always serving the best single arm? None when it cannot be told."""
    pol = b.get("policies") or {}
    learners = [pol.get(k) or {} for k in ("linucb", "epsilon_greedy")]
    fixed = b.get("best_fixed_arm") or {}
    regrets = [float(p["cum_regret"]) for p in learners if p.get("cum_regret") is not None]
    if fixed.get("cum_regret") is not None and regrets:
        return min(regrets) < float(fixed["cum_regret"])
    best = _best_arm(b)
    rewards = [float(p["mean_reward"]) for p in learners if p.get("mean_reward") is not None]
    if best is not None and rewards:
        return max(rewards) >= float(best[1])
    return None


def _ndcg(m: Any) -> Any:
    return (m or {}).get("ndcg_at_10")


def _ranking_headlines(rm: dict) -> list[str]:
    """Top-10 headline: item-kNN and MF + prior vs popularity (warm test users, then simulated new users)."""
    out = []
    pop = _ndcg(rm.get("popularity"))
    rivals = [("item-kNN", _ndcg(rm.get("item_knn"))), ("MF + popularity prior", _ndcg(rm.get("cf_mf_pop")))]
    if pop is not None and any(v is not None for _, v in rivals):
        rows = rivals + [("popularity", pop), ("MF ranked by its predicted rating", _ndcg(rm.get("cf_mf")))]
        cells = ", ".join(f"{name} {fmt(v)}" for name, v in rows if v is not None)
        out.append(f"**Top-10 ranking** (warm movies): NDCG@10 {cells}.")
    nu = rm.get("new_users") or {}
    strategies = nu.get("strategies") or {}
    knn, npop = _ndcg(strategies.get("item_knn")), _ndcg(strategies.get("popularity"))
    if knn is not None and npop is not None:
        out.append(f"**New users** (simulated, {nu.get('n_seed_ratings', 'n/a')} known ratings): item-kNN NDCG@10 "
                   f"{fmt(knn)} vs {fmt(npop)} for popularity.")
    return out


def _headlines(metrics: dict) -> list[str]:
    """The key numbers a reader should see first (each only when its inputs exist)."""
    rm = metrics.get("rating_models") or {}
    out = []
    mf, bias = rm.get("cf_mf") or {}, rm.get("bias_baseline") or {}
    if mf.get("rmse") is not None and bias.get("rmse") is not None:
        out.append(f"**Rating prediction:** matrix factorization test RMSE {fmt(mf['rmse'])} vs {fmt(bias['rmse'])} "
                   "for the bias baseline (lower is better).")
    out += _ranking_headlines(rm)
    cold = (rm.get("two_tower") or {}).get("cold") or {}
    baselines = rm.get("cold_baselines") or {}
    rand, genre_pop = baselines.get("random") or {}, baselines.get("genre_popularity") or {}
    if cold.get("ndcg_at_10") is not None and rand.get("ndcg_at_10") is not None:
        text = (f"**Brand-new movies:** the two-tower model ranks them with NDCG@10 {fmt(cold['ndcg_at_10'])} vs "
                f"{fmt(rand['ndcg_at_10'])} for random ranking")
        if genre_pop.get("ndcg_at_10") is not None:
            text += f" and {fmt(genre_pop['ndcg_at_10'])} for a non-personalized genre-popularity ranking"
        out.append(text + ".")
    a = metrics.get("anomaly") or {}
    if a.get("test_f1") is not None:
        per = {t: r for t, r in (a.get("per_bot_type_recall") or {}).items() if r is not None}
        missed = sum(float(r) < MISSED_RECALL for r in per.values())
        miss_text = f"; misses {missed} of {len(per)} bot types" if per else ""
        out.append(f"**Bot detection:** test F1 {fmt(a['test_f1'])} on synthetic bots (precision "
                   f"{fmt(a.get('test_precision'))}, recall {fmt(a.get('test_recall'))}{miss_text}).")
    b = metrics.get("bandit") or {}
    pol = b.get("policies") or {}
    lin, rnd = pol.get("linucb") or {}, pol.get("random") or {}
    if lin.get("cum_regret") is not None and rnd.get("cum_regret"):
        drop = 100 * (1 - float(lin["cum_regret"]) / float(rnd["cum_regret"]))
        text = (f"**Contextual bandit:** LinUCB's cumulative regret is {drop:.0f}% below random "
                f"({fmt(lin['cum_regret'])} vs {fmt(rnd['cum_regret'])}, simulated clicks)")
        if _context_pays_off(b) is False:
            text += ", but it does not beat always serving the best single strategy"
        out.append(text + ".")
    if not out:
        return []
    return ["**Headline results** (held-out data):", ""] + [f"- {t}" for t in out] + [""]


def render_results_markdown(metrics: dict) -> str:
    """Render the full RESULTS block (markdown, without the markers)."""
    lines: list[str] = []
    if metrics.get("fast"):
        lines += ["> **FAST MODE — numbers not final.** Generated with `--fast`; rerun the full pipeline.", ""]
    lines += _headlines(metrics)
    lines += [_data_sentence(metrics), ""]
    rm = metrics.get("rating_models") or {}
    lines += _rating_section(rm)
    new_users = _new_user_section(rm)
    if new_users:
        lines += [""] + new_users
    lines += [""] + _cold_section(rm)
    lines += [""] + _anomaly_section(metrics.get("anomaly")) + [""]
    lines += [_clustering_line(metrics.get("clustering")), ""]
    lines += _bandit_section(metrics.get("bandit"))
    lines += [""] + _dqn_section(metrics.get("dqn")) + [""]

    details = _anomaly_details(metrics.get("anomaly")) + _bandit_details(metrics.get("bandit"))
    if details:
        lines += ["<details>", "<summary>Anomaly and bandit details</summary>", ""] + details + ["", "</details>", ""]

    seconds = metrics.get("train_seconds") or {}
    done = {k: v for k, v in seconds.items() if v is not None}
    if done:
        steps = ", ".join(f"{k} {float(v):.0f}s" for k, v in done.items())
        lines += [f"Training time: {sum(float(v) for v in done.values()):.0f}s, the sum of the step times ({steps}).",
                  ""]

    figs = _figures(metrics)
    if figs:
        lines += ["**Key figures:**", "", "<p>"] + _figure_grid(figs) + ["</p>", ""]
    lines.append(f"_Generated by `scripts/evaluate.py` at {metrics.get('generated_at', 'n/a')} "
                 f"from `reports/metrics.json`._")
    return "\n".join(lines)


def split_readme(text: str) -> tuple[str, str, str]:
    """Split README text into (head incl. start marker, block, tail incl. end marker).

    Only markers standing alone on their own line count, so the marker names may be
    mentioned inline elsewhere in the README. Raises ``ValueError`` if missing.
    """
    start = re.search(rf"^{re.escape(MARKER_START)}[ \t]*$", text, flags=re.MULTILINE)
    end = re.search(rf"^{re.escape(MARKER_END)}[ \t]*$", text[start.end():], flags=re.MULTILINE) if start else None
    if start is None or end is None:
        raise ValueError(f"README must contain a line {MARKER_START} followed by a line {MARKER_END}")
    end_pos = start.end() + end.start()
    return text[: start.end()], text[start.end(): end_pos], text[end_pos:]


def update_readme(readme_path: Path | str, block: str) -> None:
    """Replace the text between the RESULTS markers with ``block`` (idempotent).

    Raises ``ValueError`` if the markers are missing or out of order.
    """
    path = Path(readme_path)
    head, _, tail = split_readme(path.read_text(encoding="utf-8"))
    path.write_text(f"{head}\n{block.strip()}\n{tail}", encoding="utf-8", newline="\n")
