"""Evaluate every trained model, write reports/metrics.json and refresh the README.

Usage (from the repo root):
    python scripts/evaluate.py          # full evaluation
    python scripts/evaluate.py --fast   # smoke mode (README gets a FAST MODE banner)

Each module ``cinematch.<step>`` exposes ``eval_step(cfg, ctx)``; missing modules or
missing model artifacts leave ``null`` in metrics.json instead of a fake number.
"""
import _bootstrap  # noqa: F401  (adds src/ to sys.path)

import argparse
import json
from pathlib import Path

from cinematch.config import MODELS_DIR, README_PATH, REPORTS_DIR, ensure_dirs, get_config
from cinematch.readme_fill import assemble_metrics, render_results_markdown, update_readme
from train_all import run

# data first: its eval_step fills ctx["data"] (processed artifacts, eval users, holdouts).
EVAL_ORDER = ["data", "cf", "content", "clustering", "anomaly", "bandit", "dqn"]


def load_train_seconds(path: Path | None = None) -> dict:
    """Per-step training times written by train_all.py ({} if training never ran)."""
    path = path or MODELS_DIR / "train_timings.json"  # resolved at call time (tests patch MODELS_DIR)
    if not path.exists():
        print(f"[evaluate] {path} not found; train_seconds will be null")
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="Evaluate CineMatch models and update the README.")
    parser.add_argument("--fast", action="store_true",
                        help="smoke mode: smaller eval samples (numbers are not final)")
    args = parser.parse_args(argv)

    cfg = get_config(fast=True if args.fast else None)
    ensure_dirs()
    print(f"[evaluate] fast={cfg.fast} seed={cfg.seed}")
    ctx = run(EVAL_ORDER, "eval_step", cfg)

    timings = load_train_seconds()
    metrics = assemble_metrics(ctx["outputs"], cfg, timings)
    if timings.get("fast") and not cfg.fast:
        # Models came from `train_all.py --fast`: the numbers are not final either.
        print("[evaluate] WARNING: models were trained with --fast; marking results as FAST MODE")
        metrics["fast"] = True
    ctx["metrics"] = metrics
    out_path = REPORTS_DIR / "metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[evaluate] wrote {out_path}")

    update_readme(README_PATH, render_results_markdown(metrics))
    print(f"[evaluate] updated results block in {README_PATH}")
    return metrics


if __name__ == "__main__":
    main()
