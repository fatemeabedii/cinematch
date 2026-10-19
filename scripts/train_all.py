"""Train every CineMatch model in the fixed pipeline order.

Usage (from the repo root):
    python scripts/train_all.py          # full run (real artifacts)
    python scripts/train_all.py --fast   # smoke run, tiny epochs (never commit its numbers)

Each module ``cinematch.<step>`` exposes ``train_step(cfg, ctx)``. A missing module or
function is an error (non-zero exit), so a typo in the step list can never pass silently.
"""
import _bootstrap  # noqa: F401  (adds src/ to sys.path)

import argparse
import importlib
import json
import time

from cinematch.config import MODELS_DIR, ensure_dirs, get_config, set_seed

# Fixed: anomaly before data (flagged ids), data before models, bandit needs
# cf/content/clustering (via Recommender), dqn needs the bandit env.
TRAIN_ORDER = ["anomaly", "data", "cf", "content", "clustering", "bandit", "dqn"]


def run(order: list[str], fn_name: str, cfg, ctx: dict | None = None) -> dict:
    """Run ``cinematch.<name>.<fn_name>(cfg, ctx)`` for each step in ``order``.

    Each step's return value is stored in ``ctx["outputs"][name]`` and its wall time (s) in
    ``ctx["timings"][name]``. A missing module or function, or an exception raised by a step,
    propagates (non-zero exit code).
    """
    if ctx is None:
        ctx = {}
    ctx.setdefault("flagged_user_ids", [])
    ctx.setdefault("timings", {})
    ctx.setdefault("outputs", {})
    for name in order:
        fn = getattr(importlib.import_module(f"cinematch.{name}"), fn_name)
        set_seed(cfg.seed)
        t0 = time.time()
        ctx["outputs"][name] = fn(cfg, ctx)
        ctx["timings"][name] = round(time.time() - t0, 1)
        print(f"[done] {name}.{fn_name} in {ctx['timings'][name]}s")
    return ctx


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="Train all CineMatch models.")
    parser.add_argument("--fast", action="store_true",
                        help="smoke mode: tiny epochs/samples (numbers are not final)")
    args = parser.parse_args(argv)

    cfg = get_config(fast=True if args.fast else None)
    ensure_dirs()
    print(f"[train_all] fast={cfg.fast} seed={cfg.seed}")
    t0 = time.time()
    ctx = run(TRAIN_ORDER, "train_step", cfg)

    timings = dict(ctx["timings"])
    timings["total"] = round(time.time() - t0, 1)
    timings["fast"] = cfg.fast  # provenance: evaluate.py adds the FAST banner if training was fast
    path = MODELS_DIR / "train_timings.json"
    path.write_text(json.dumps(timings, indent=2), encoding="utf-8")
    print(f"[train_all] total {timings['total']}s -> {path}")
    return ctx


if __name__ == "__main__":
    main()
