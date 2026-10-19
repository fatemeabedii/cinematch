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
POLICY_LABELS = {
    "random": "Random",
    "epsilon_greedy": "Epsilon-greedy",
    "linucb": "LinUCB",
}


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
    """Build the metrics.json dict (section 6.3) from ``{step_name: eval_step output}``.

    Assembly rule: ``cf``/``content`` outputs are merged into ``rating_models``; every
    other step's output is stored under its own name. Steps that were skipped (absent
    from ``step_outputs``) or returned ``None`` leave ``null``.
    """
    seconds = {name: None for name in TRAIN_STEPS}
    for name, value in (train_seconds or {}).items():
        if name in seconds:
            seconds[name] = value

    cold_pct = round(100 * getattr(cfg, "cold_item_frac", 0.06))
    metrics: dict[str, Any] = {
        "generated_at": generated_at or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": "MovieLens-1M",
        "split": f"per-user time-aware 80/10/10, {cold_pct}% cold items, seed {getattr(cfg, 'seed', 42)}",
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


def _data_sentence(metrics: dict) -> str:
    d = metrics.get("data")
    split = metrics.get("split", "")
    if not d:
        return f"**Data:** {metrics.get('dataset', 'MovieLens-1M')} ({split}); split statistics not available."
    return (
        f"**Data:** {metrics.get('dataset', 'MovieLens-1M')}, {_count(d.get('n_users'))} users, "
        f"{_count(d.get('n_items'))} movies ({_count(d.get('n_cold_items'))} held out as cold-start items). "
        f"Split: {split}: {_count(d.get('n_train'))} train / {_count(d.get('n_val'))} validation / "
        f"{_count(d.get('n_test'))} test / {_count(d.get('n_cold'))} cold ratings. "
        f"{_count(d.get('n_flagged_users'))} users flagged by anomaly detection are excluded from training "
        f"and evaluation. Ranking metrics use a fixed seeded sample of {_count(d.get('n_eval_users'))} test users; "
        f"a held-out rating >= 4 counts as relevant."
    )


def _rating_section(rm: dict) -> list[str]:
    lines = [
        "| Model | RMSE | P@10 | R@10 | NDCG@10 |",
        "|---|---|---|---|---|",
    ]
    for key, label in MODEL_LABELS.items():
        lines.append(_model_row(label, rm.get(key)))
    tt = rm.get("two_tower") or {}
    cold = tt.get("cold")
    lines.append(_model_row("Two-tower on **cold-start** movies", cold))
    lines.append("")
    notes = []
    mf = rm.get("cf_mf")
    if mf:
        notes.append(f"MF: dim {mf.get('dim', 'n/a')}, best epoch {mf.get('best_epoch', 'n/a')} (early stopping on validation RMSE).")
    if cold:
        notes.append(
            f"Cold-start row: movies never seen in training, scored from metadata only "
            f"({_count(cold.get('n_eval_users'))} users with a relevant cold rating); MF cannot score them by construction."
        )
    notes.append("Popularity has no rating prediction (RMSE n/a); it may beat MF on top-N, reported as measured.")
    lines.extend(f"- {n}" for n in notes)
    return lines


def _anomaly_line(a: dict | None) -> str:
    if not a:
        return "**Anomaly detection:** n/a (not run)."
    per_type = a.get("per_bot_type_recall") or {}
    types = ", ".join(f"{name} {fmt(r)}" for name, r in per_type.items())
    capped = " (capped at 3% of real users)" if a.get("epsilon_capped") else ""
    line = (
        f"**Anomaly detection:** Gaussian density, epsilon {fmt(a.get('epsilon'))}{capped} chosen by validation F1 "
        f"({fmt(a.get('val_f1'))}). Test set ({_count(a.get('n_bots_test'))} synthetic bots + held-out real users): "
        f"precision {fmt(a.get('test_precision'))}, recall {fmt(a.get('test_recall'))}, F1 {fmt(a.get('test_f1'))}. "
        f"Real users flagged and excluded: {_count(a.get('n_real_flagged_all'))}."
    )
    if types:
        line += f" Recall per bot type: {types}."
    return line


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
    lines.append("")
    arms = b.get("arm_mean_expected_reward") or {}
    arm_text = ", ".join(f"{name} {fmt(v)}" for name, v in arms.items())
    lines.append(f"Oracle mean reward {fmt(b.get('oracle_mean_reward'))}; mean expected reward per arm: {arm_text}.")
    return lines


def _dqn_line(d: dict | None) -> str:
    if not d:
        return "**DQN stretch goal:** not shipped."
    return (
        f"**DQN stretch goal:** mean episode reward (last 50 episodes) {fmt(d.get('mean_reward_last50'))} "
        f"vs random order {fmt(d.get('random'))} and greedy-by-CF {fmt(d.get('greedy'))}."
    )


def _figures(metrics: dict) -> list[str]:
    """Key figures, linked only when the step that draws them produced metrics."""
    rm = metrics.get("rating_models") or {}
    figs = []
    if rm.get("cf_mf"):
        figs.append(("MF training curve", "cf_training.png"))
    if metrics.get("clustering"):
        figs.append(("Taste profiles (PCA of MF user vectors)", "clusters_pca.png"))
    if metrics.get("anomaly"):
        figs.append(("Anomaly scores: real users vs bots", "anomaly_scores.png"))
    if metrics.get("bandit"):
        figs.append(("Bandit cumulative regret", "bandit_regret.png"))
    return [f"![{alt}](reports/figures/{name})" for alt, name in figs]


def render_results_markdown(metrics: dict) -> str:
    """Render the full RESULTS block (markdown, without the markers)."""
    lines: list[str] = []
    if metrics.get("fast"):
        lines += ["> **FAST MODE — numbers not final.** Generated with `--fast`; rerun the full pipeline.", ""]
    lines += [_data_sentence(metrics), ""]
    lines += _rating_section(metrics.get("rating_models") or {})
    lines += ["", _anomaly_line(metrics.get("anomaly")), ""]
    lines += [_clustering_line(metrics.get("clustering")), ""]
    lines += _bandit_section(metrics.get("bandit"))
    lines += ["", _dqn_line(metrics.get("dqn")), ""]

    seconds = metrics.get("train_seconds") or {}
    done = {k: v for k, v in seconds.items() if v is not None}
    if done:
        steps = ", ".join(f"{k} {float(v):.0f}s" for k, v in done.items())
        lines += [f"Training time: {sum(float(v) for v in done.values()):.0f}s total ({steps}).", ""]

    figs = _figures(metrics)
    if figs:
        lines += figs + [""]
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
