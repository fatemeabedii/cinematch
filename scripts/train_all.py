"""Train every CineMatch model in the fixed pipeline order.

Usage (from the repo root):
    python scripts/train_all.py          # full run (real artifacts)
    python scripts/train_all.py --fast   # smoke run, tiny epochs (never commit its numbers)

Each module ``cinematch.<step>`` exposes ``train_step(cfg, ctx)``; modules that do not
exist yet are skipped, so the pipeline runs at every stage of development.
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

    Missing modules/functions are skipped with a message. Each step's return value is
    stored in ``ctx["outputs"][name]`` and its wall time (s) in ``ctx["timings"][name]``.
    Exceptions raised by a step propagate (non-zero exit code).
    """
    if ctx is None:
        ctx = {}
    ctx.setdefault("flagged_user_ids", [])
    ctx.setdefault("timings", {})
    ctx.setdefault("outputs", {})
    for name in order:
        module_name = f"cinematch.{name}"
        try:
            mod = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            if exc.name != module_name:
                raise  # the module exists but one of ITS imports is broken
            print(f"[skip] {module_name} not implemented yet")
            continue
        fn = getattr(mod, fn_name, None)
        if fn is None:
            print(f"[skip] {module_name}.{fn_name} missing")
            continue
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
