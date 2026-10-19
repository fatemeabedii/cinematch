"""Contextual bandit: which recommendation strategy should serve this user's list?

Arms are the four Recommender strategies (`ARMS`). Each round a precomputed user arrives, a policy
picks an arm, the arm's top-10 list is shown, and a click simulator built from the user's TEST
ratings draws one Bernoulli click per item: p(click | test rating r) = cfg.click_probs[r], items
without a test rating get cfg.click_unrated (validation ratings are never used: they tuned early
stopping). Reward = fraction of the 10 items clicked. Because every per-item probability is known,
the expected reward of every arm for every user is exact, and so is the regret.

Policies: uniform random, epsilon-greedy (per-arm running mean, ignores context) and disjoint
LinUCB on the user context [1, log1p(train count), mean rating, rating std, one_hot(cluster)].
LinUCB's alpha and epsilon-greedy's epsilon are tuned (lowest mean cumulative regret over a small
grid) on a separate simulator built from users DISJOINT from the reported ones (not flagged, not in
eval_users.npy) with separate seeds; the reported run on eval users then uses the chosen values.

Module import is numpy-only (the app imports it); `cinematch.recommend` / `cinematch.data` are
imported lazily inside `train_step`, matplotlib inside `eval_step`. Torch is never used.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from cinematch.config import FIGURES_DIR, MODELS_DIR, PROCESSED_DIR

ARMS = ["cf", "content", "cluster_popular", "trending"]
LIST_LEN = 10
CURVE_STEP = 10                     # results curves are subsampled every CURVE_STEP rounds
POLICY_NAMES = ["random", "epsilon_greedy", "linucb"]
# Context columns of two_tower.npz["user_feat_raw"]: log1p(count), mean rating, rating std.
CONTEXT_FEAT_COLS = (38, 36, 37)


def _log(msg: str) -> None:
    print(f"[bandit] {msg}")


# ----------------------------------------------------------------------------- environment
class BanditEnv:
    """Click simulator over precomputed users (env dict from `precompute_env` / `load_env`).

    Every `step` draws exactly LIST_LEN uniforms and every `sample_user` one integer, whatever arm
    is pulled, so two policies run with the same seed see the same users and the same click noise
    (common random numbers: their difference is due to the policy alone).
    """

    def __init__(self, env: dict, rng: np.random.Generator):
        self.probs = np.asarray(env["probs"], dtype=np.float64)           # [n_users, n_arms, 10]
        self.expected_rewards = np.asarray(env["expected"], dtype=np.float64)  # [n_users, n_arms]
        self.contexts = np.asarray(env["contexts"], dtype=np.float64)      # [n_users, d]
        self.n_users, self.n_arms, self.list_len = self.probs.shape
        self.dim = self.contexts.shape[1]
        self.rng = rng

    def sample_user(self) -> int:
        """Uniformly random precomputed user (row index into the env arrays)."""
        return int(self.rng.integers(self.n_users))

    def context(self, u: int) -> np.ndarray:
        return self.contexts[u]

    def step(self, u: int, arm: int) -> float:
        """Show arm's list to user u: mean of one Bernoulli(p_i) click per item, in {0, .1, ..., 1}."""
        clicks = self.rng.random(self.list_len) < self.probs[u, arm]
        return float(clicks.mean())

    def expected(self, u: int) -> np.ndarray:
        """Exact expected reward of every arm for user u, shape [n_arms]."""
        return self.expected_rewards[u]


# ----------------------------------------------------------------------------- policies
def _argmax_random_tie(values: np.ndarray, rng: np.random.Generator) -> int:
    best = np.flatnonzero(values >= values.max() - 1e-12)
    return int(best[0]) if len(best) == 1 else int(rng.choice(best))


class Policy:
    """Interface: `select(x) -> arm`, then `update(x, arm, reward)` with the observed reward."""

    name = "policy"

    def __init__(self, n_arms: int, rng: np.random.Generator):
        self.n_arms = n_arms
        self.rng = rng

    def select(self, x: np.ndarray) -> int:
        raise NotImplementedError

    def update(self, x: np.ndarray, arm: int, reward: float) -> None:
        pass


class RandomPolicy(Policy):
    """Uniformly random arm (the no-learning baseline)."""

    name = "random"

    def select(self, x: np.ndarray) -> int:
        return int(self.rng.integers(self.n_arms))


class EpsilonGreedy(Policy):
    """Context-free: explore with probability epsilon, else the arm with the best running mean."""

    name = "epsilon_greedy"

    def __init__(self, n_arms: int, rng: np.random.Generator, epsilon: float = 0.1):
        super().__init__(n_arms, rng)
        self.epsilon = float(epsilon)
        self.counts = np.zeros(n_arms)
        self.means = np.zeros(n_arms)

    def select(self, x: np.ndarray) -> int:
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_arms))
        return _argmax_random_tie(self.means, self.rng)

    def update(self, x: np.ndarray, arm: int, reward: float) -> None:
        self.counts[arm] += 1
        self.means[arm] += (reward - self.means[arm]) / self.counts[arm]


class LinUCB(Policy):
    """Disjoint LinUCB: per arm A_a = ridge*I + sum x xᵀ, b_a = sum r x, θ_a = A_a⁻¹ b_a.

    Score p_a = θ_aᵀx + alpha * sqrt(xᵀ A_a⁻¹ x); A_a⁻¹ is kept up to date with Sherman-Morrison.
    """

    name = "linucb"

    def __init__(self, n_arms: int, dim: int, rng: np.random.Generator, alpha: float = 1.0,
                 ridge: float = 1.0):
        super().__init__(n_arms, rng)
        self.alpha = float(alpha)
        self.A_inv = np.repeat(np.eye(dim)[None] / float(ridge), n_arms, axis=0)   # [K, d, d]
        self.b = np.zeros((n_arms, dim))

    def scores(self, x: np.ndarray) -> np.ndarray:
        """UCB score of every arm for context x."""
        theta = np.einsum("kij,kj->ki", self.A_inv, self.b)
        Ax = self.A_inv @ x                                                  # [K, d]
        return theta @ x + self.alpha * np.sqrt(np.maximum(Ax @ x, 0.0))

    def select(self, x: np.ndarray) -> int:
        return _argmax_random_tie(self.scores(x), self.rng)

    def update(self, x: np.ndarray, arm: int, reward: float) -> None:
        Ax = self.A_inv[arm] @ x
        self.A_inv[arm] -= np.outer(Ax, Ax) / (1.0 + x @ Ax)
        self.b[arm] += reward * x


def make_policies(cfg, n_arms: int, dim: int, seed: int) -> dict[str, Policy]:
    """The three policies of the comparison, each with its own seeded generator."""
    return {
        "random": RandomPolicy(n_arms, np.random.default_rng([seed, 1])),
        "epsilon_greedy": EpsilonGreedy(n_arms, np.random.default_rng([seed, 2]), cfg.bandit_epsilon),
        "linucb": LinUCB(n_arms, dim, np.random.default_rng([seed, 3]), cfg.bandit_alpha, cfg.bandit_ridge),
    }


# ----------------------------------------------------------------------------- simulation
def run_simulation(env: BanditEnv, policy: Policy, rounds: int, rng: np.random.Generator | None = None) -> dict:
    """Play `rounds` rounds. `rng` (if given) replaces env.rng and drives users + clicks.

    Returns arrays of length `rounds`: "reward" (observed), "regret" (expected best - expected
    chosen, exact), "arms" (chosen arm) and "oracle" (expected reward of the best arm).
    """
    if rng is not None:
        env.rng = rng
    reward = np.zeros(rounds)
    regret = np.zeros(rounds)
    oracle = np.zeros(rounds)
    arms = np.zeros(rounds, dtype=np.int64)
    for t in range(rounds):
        u = env.sample_user()
        x = env.context(u)
        arm = policy.select(x)
        r = env.step(u, arm)
        policy.update(x, arm, r)
        exp_u = env.expected(u)
        reward[t], arms[t] = r, arm
        oracle[t] = exp_u.max()
        regret[t] = oracle[t] - exp_u[arm]
    return {"reward": reward, "regret": regret, "arms": arms, "oracle": oracle}


def _curve_idx(rounds: int, step: int = CURVE_STEP) -> np.ndarray:
    """Rounds (0-based) kept in the subsampled curves: every `step`-th, plus the last one."""
    idx = np.arange(step - 1, rounds, step)
    if len(idx) == 0 or idx[-1] != rounds - 1:
        idx = np.append(idx, rounds - 1)
    return idx


def _r4(values):
    return [round(float(v), 4) for v in values]


def _seed_std(values: np.ndarray, axis: int = 0) -> np.ndarray:
    """Sample std (ddof=1) across seeds; 0 when there is a single seed."""
    values = np.asarray(values, dtype=np.float64)
    if values.shape[axis] < 2:
        return np.zeros_like(values.mean(axis=axis))
    return values.std(axis=axis, ddof=1)


def frac_users_no_signal(expected: np.ndarray) -> float:
    """Fraction of users whose 4 arms have identical expected reward (regret 0 whatever is chosen).

    This happens when none of the user's shown items has a test rating (every slot is click_unrated);
    only the remaining users carry any signal about which arm is better.
    """
    expected = np.asarray(expected, dtype=np.float64)
    return float(np.mean(expected.max(axis=1) - expected.min(axis=1) < 1e-9))


def run_all(env_dict: dict, cfg) -> dict:
    """Every policy x every seed in cfg.bandit_seeds for cfg.bandit_rounds rounds.

    Seed s gives every policy the same user sequence and click noise. Returns (spec 7.2) per policy
    mean reward, final cumulative regret mean/std over seeds, mean arm counts and mean cumulative
    reward/regret curves (+ regret std), subsampled every CURVE_STEP rounds; plus the oracle mean
    reward, every arm's mean expected reward over the precomputed users, rounds, n_seeds, the
    fraction of users with no arm signal and the policy hyperparameters. Stds are sample stds
    (ddof=1) across seeds.
    """
    rounds, seeds = int(cfg.bandit_rounds), list(cfg.bandit_seeds)
    n_arms = np.asarray(env_dict["probs"]).shape[1]
    dim = np.asarray(env_dict["contexts"]).shape[1]
    arm_names = [str(a) for a in env_dict.get("arm_names", ARMS)]
    runs = {name: [] for name in POLICY_NAMES}
    oracle_means = []
    for seed in seeds:
        for name, policy in make_policies(cfg, n_arms, dim, seed).items():
            env = BanditEnv(env_dict, np.random.default_rng(seed))
            runs[name].append(run_simulation(env, policy, rounds))
        oracle_means.append(runs[POLICY_NAMES[0]][-1]["oracle"].mean())

    idx = _curve_idx(rounds)
    policies = {}
    for name, sims in runs.items():
        cum_reward = np.cumsum([s["reward"] for s in sims], axis=1)          # [n_seeds, T]
        cum_regret = np.cumsum([s["regret"] for s in sims], axis=1)
        counts = np.mean([np.bincount(s["arms"], minlength=n_arms) for s in sims], axis=0)
        policies[name] = {
            "mean_reward": round(float(np.mean([s["reward"].mean() for s in sims])), 4),
            "cum_regret": round(float(cum_regret[:, -1].mean()), 4),
            "cum_regret_std": round(float(_seed_std(cum_regret[:, -1])), 4),
            "arm_counts": _r4(counts),
            "cum_reward_curve": _r4(cum_reward.mean(axis=0)[idx]),
            "cum_regret_curve": _r4(cum_regret.mean(axis=0)[idx]),
            "cum_regret_curve_std": _r4(_seed_std(cum_regret, axis=0)[idx]),
        }
    expected = np.asarray(env_dict["expected"], dtype=np.float64)
    return {
        "policies": policies,
        "oracle_mean_reward": round(float(np.mean(oracle_means)), 4),
        "arm_mean_expected_reward": {a: round(float(v), 4) for a, v in zip(arm_names, expected.mean(axis=0))},
        "rounds": rounds,
        "n_seeds": len(seeds),
        "frac_users_no_signal": round(frac_users_no_signal(expected), 4),
        "params": {"epsilon": float(cfg.bandit_epsilon), "alpha": float(cfg.bandit_alpha),
                   "ridge": float(cfg.bandit_ridge)},
    }


def _tune(env_dict: dict, cfg, make_policy, values, seeds) -> dict[float, dict]:
    """Final cumulative regret (mean, sample std over `seeds`) of `make_policy(value, rng)` per value."""
    rounds = int(cfg.bandit_rounds)
    out = {}
    for value in values:
        final = []
        for seed in seeds:
            sim = run_simulation(BanditEnv(env_dict, np.random.default_rng(seed)),
                                 make_policy(value, seed), rounds)
            final.append(sim["regret"].sum())
        out[float(value)] = {"cum_regret": round(float(np.mean(final)), 4),
                             "cum_regret_std": round(float(_seed_std(final)), 4)}
    return out


def tune_alpha(env_dict: dict, cfg, alphas, seeds) -> dict[float, dict]:
    """LinUCB final cumulative regret (mean, sample std over `seeds`) for every alpha in `alphas`.

    Meant to be run on a TUNING env: users disjoint from the reported ones (see `tuning_user_pool`)
    and seeds disjoint from cfg.bandit_seeds, so alpha is never picked on the reported run.
    """
    n_arms = np.asarray(env_dict["probs"]).shape[1]
    dim = np.asarray(env_dict["contexts"]).shape[1]
    return _tune(env_dict, cfg, lambda a, s: LinUCB(n_arms, dim, np.random.default_rng([s, 3]), a, cfg.bandit_ridge),
                 alphas, seeds)


def tune_epsilon(env_dict: dict, cfg, epsilons, seeds) -> dict[float, dict]:
    """Epsilon-greedy final cumulative regret (mean, sample std over `seeds`) for every epsilon."""
    n_arms = np.asarray(env_dict["probs"]).shape[1]
    return _tune(env_dict, cfg, lambda e, s: EpsilonGreedy(n_arms, np.random.default_rng([s, 2]), e),
                 epsilons, seeds)


def _best(table: dict[float, dict]) -> float:
    """Grid value with the lowest mean cumulative regret (exact ties: the earlier grid value)."""
    return min(table, key=lambda v: table[v]["cum_regret"])


def tune_params(tune_env: dict, cfg) -> tuple[dict, dict, dict]:
    """Pick LinUCB alpha and epsilon-greedy epsilon by lowest mean cumulative regret on `tune_env`.

    Runs cfg.bandit_rounds rounds per grid value and seed in cfg.bandit_tune_seeds. Returns
    (params {"epsilon", "alpha"}, tuning {"alpha": {"<a>": regret}, "epsilon": {"<e>": regret}},
    tuning_std: same layout with the sample std of the final regret over the tune seeds, so a
    reader can see when the winning value is within seed noise of the runner-up).
    """
    seeds = list(cfg.bandit_tune_seeds)
    tables = {"alpha": tune_alpha(tune_env, cfg, cfg.bandit_alpha_grid, seeds),
              "epsilon": tune_epsilon(tune_env, cfg, cfg.bandit_epsilon_grid, seeds)}
    params = {"epsilon": _best(tables["epsilon"]), "alpha": _best(tables["alpha"])}
    tuning = {k: {str(v): row["cum_regret"] for v, row in t.items()} for k, t in tables.items()}
    tuning_std = {k: {str(v): row["cum_regret_std"] for v, row in t.items()} for k, t in tables.items()}
    return params, tuning, tuning_std


# ----------------------------------------------------------------------------- precompute
def click_table(cfg) -> np.ndarray:
    """p(click) indexed by integer test rating: index 0 = unrated, 1..5 = cfg.click_probs."""
    table = np.full(6, float(cfg.click_unrated))
    for r, p in cfg.click_probs.items():
        table[int(r)] = float(p)
    return table


def build_contexts(user_feat_raw: np.ndarray, labels: np.ndarray, n_clusters: int) -> np.ndarray:
    """Context rows [1, log1p_count, mean_rating, std_rating, one_hot(cluster)] for the given users.

    `user_feat_raw` / `labels` are already restricted to the sampled users; the three scalars are
    standardised over those users (a constant column becomes 0).
    """
    feats = np.asarray(user_feat_raw, dtype=np.float64)[:, list(CONTEXT_FEAT_COLS)]
    std = feats.std(axis=0)
    feats = (feats - feats.mean(axis=0)) / np.where(std > 1e-12, std, 1.0)
    one_hot = np.eye(n_clusters)[np.asarray(labels, dtype=np.int64)]
    return np.hstack([np.ones((len(feats), 1)), feats, one_hot]).astype(np.float32)


def precompute_env(cfg, recommender, user_idx: np.ndarray, holdout_test: dict, user_feat_raw: np.ndarray,
                   labels: np.ndarray, n_clusters: int) -> dict:
    """Top-10 list, per-item click probabilities and exact expected reward for every (user, arm).

    `recommender` needs only `recommend_strategy(u, arm, n)` -> item indices. `user_feat_raw` and
    `labels` are indexed by user index (full arrays). A list shorter than 10 is padded with item -1
    and click probability 0 (nothing shown in that slot).
    """
    user_idx = np.asarray(user_idx, dtype=np.int64)
    n = len(user_idx)
    lists = np.full((n, len(ARMS), LIST_LEN), -1, dtype=np.int32)
    for k, u in enumerate(user_idx.tolist()):
        for a, arm in enumerate(ARMS):
            recs = np.asarray(recommender.recommend_strategy(u, arm, n=LIST_LEN), dtype=np.int64)[:LIST_LEN]
            lists[k, a, :len(recs)] = recs
    # integer test rating of every shown item (0 = no test rating), then a vectorised table lookup
    rating = np.zeros(lists.shape, dtype=np.int64)
    for k, u in enumerate(user_idx.tolist()):
        held = holdout_test.get(u)
        if held:
            rating[k] = [[int(round(held.get(i, 0))) for i in row] for row in lists[k].tolist()]
    probs = click_table(cfg)[rating]
    probs[lists < 0] = 0.0
    probs = probs.astype(np.float32)
    contexts = build_contexts(np.asarray(user_feat_raw)[user_idx], np.asarray(labels)[user_idx], n_clusters)
    return {"user_idx": user_idx, "lists": lists, "probs": probs, "expected": probs.mean(axis=2),
            "contexts": contexts, "arm_names": np.array(ARMS, dtype="U20")}


def save_env(env: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{k: np.asarray(v) for k, v in env.items()})


def load_env(models_dir: Path = MODELS_DIR) -> dict:
    """Arrays of models/bandit_env.npz (numpy only; FileNotFoundError if the bandit never ran)."""
    path = Path(models_dir) / "bandit_env.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python scripts/train_all.py")
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def sample_users(users: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Sorted seeded sample of n users from `users` (all of them if fewer).

    Disjointness of the tuning users is enforced by `tuning_user_pool`, not here.
    """
    pool = np.asarray(users)
    n = min(int(n), len(pool))
    return np.sort(np.random.default_rng(seed).choice(pool, size=n, replace=False))


def tuning_user_pool(holdout_test: dict, eval_users: np.ndarray, flagged=()) -> np.ndarray:
    """Sorted users with >= 1 test rating that are neither flagged nor in `eval_users`.

    The bandit hyperparameters are tuned on (a sample of) these users, so selection never sees the
    users of the reported simulation (which are drawn from eval_users).
    """
    users = np.array(sorted(u for u, held in holdout_test.items() if held), dtype=np.int64)
    drop = np.union1d(np.asarray(eval_users, dtype=np.int64), np.asarray(flagged, dtype=np.int64))
    return users[~np.isin(users, drop)]


def _dirs(ctx: dict) -> tuple[Path, Path, Path]:
    return (Path(ctx.get("models_dir", MODELS_DIR)), Path(ctx.get("processed_dir", PROCESSED_DIR)),
            Path(ctx.get("figures_dir", FIGURES_DIR)))


def _summary(results: dict) -> dict:
    """The metrics.json "bandit" dict: run_all results without the curves."""
    out = {k: v for k, v in results.items() if k != "policies"}
    out["policies"] = {name: {k: v for k, v in p.items() if "_curve" not in k}
                       for name, p in results["policies"].items()}
    return out


def _flagged_indices(data: dict, processed_dir: Path) -> np.ndarray:
    """User indices flagged by anomaly detection (from data["index_maps"] or processed index_maps.json)."""
    index_maps = data.get("index_maps")
    if index_maps is None and (processed_dir / "index_maps.json").exists():
        index_maps = json.loads((processed_dir / "index_maps.json").read_text(encoding="utf-8"))
    if index_maps is None:
        _log("index_maps not found: no flagged users excluded from tuning")
        return np.zeros(0, dtype=np.int64)
    from cinematch.data import flagged_indices
    return flagged_indices(index_maps)


def train_step(cfg, ctx: dict) -> dict:
    """Precompute the simulators, tune alpha/epsilon on disjoint users, run every policy, write results.

    Reported users: cfg.bandit_users sampled (cfg.seed) from models/eval_users.npy. Tuning users:
    cfg.bandit_tune_users sampled (cfg.seed) from `tuning_user_pool` (>= 1 test row, not flagged, not
    in eval_users.npy); alpha and epsilon are picked there over cfg.bandit_tune_seeds (`tune_params`)
    and then used for the reported run on the eval-user env with cfg.bandit_seeds.
    Clicks use TEST ratings only (ctx["data"]["holdout_test"] if present, else loaded from processed
    ratings_split). `ctx["recommender"]` may supply any object with `recommend_strategy` (test hook);
    by default a `cinematch.recommend.Recommender(models_dir, processed_dir)` is built.
    Writes models/bandit_env.npz (eval users only) and models/bandit_results.json (incl. "tuning" and
    "tuning_std": per grid value, mean / seed std of the tuning regret); returns the metrics
    "bandit" dict.
    """
    models_dir, processed_dir, _ = _dirs(ctx)
    recommender = ctx.get("recommender")
    if recommender is None:
        from cinematch.recommend import Recommender
        recommender = Recommender(models_dir=models_dir, processed_dir=processed_dir)
    data = ctx.get("data") or {}
    holdout = data.get("holdout_test")
    if holdout is None:
        from cinematch.data import holdout_dict, load_processed
        data = load_processed(processed_dir)
        holdout = holdout_dict(data["ratings"], "test")

    eval_users = np.load(models_dir / "eval_users.npy")
    user_idx = sample_users(eval_users, cfg.bandit_users, cfg.seed)
    pool = tuning_user_pool(holdout, eval_users, _flagged_indices(data, processed_dir))
    tune_idx = sample_users(pool, cfg.bandit_tune_users, cfg.seed)
    with np.load(models_dir / "two_tower.npz", allow_pickle=False) as z:
        user_feat_raw = z["user_feat_raw"]
    with np.load(models_dir / "clusters.npz", allow_pickle=False) as z:
        labels = z["labels"]
        n_clusters = int(z["centroids"].shape[0]) if "centroids" in z.files else int(labels.max()) + 1

    def build(users):
        return precompute_env(cfg, recommender, users, holdout, user_feat_raw, labels, n_clusters)

    env = build(user_idx)
    save_env(env, models_dir / "bandit_env.npz")
    _log(f"env: {len(user_idx)} users x {len(ARMS)} arms, context dim {env['contexts'].shape[1]}; "
         f"arm mean expected reward {np.round(env['expected'].mean(axis=0), 4).tolist()}; "
         f"users with no arm signal {frac_users_no_signal(env['expected']):.1%}")

    run_cfg = copy.copy(cfg)                     # never dataclasses.replace: __post_init__ would re-apply FAST
    if len(tune_idx):
        tune_env = build(tune_idx)
        params, tuning, tuning_std = tune_params(tune_env, cfg)
        run_cfg.bandit_alpha, run_cfg.bandit_epsilon = params["alpha"], params["epsilon"]
        _log(f"tuning on {len(tune_idx)} disjoint users (pool {len(pool)}, no signal "
             f"{frac_users_no_signal(tune_env['expected']):.1%}): alpha regret {tuning['alpha']}, "
             f"epsilon regret {tuning['epsilon']} -> alpha={params['alpha']}, epsilon={params['epsilon']}")
    else:
        tuning, tuning_std = {"alpha": {}, "epsilon": {}}, {"alpha": {}, "epsilon": {}}
        _log("no tuning users available: using config alpha/epsilon")

    results = run_all(env, run_cfg)
    results["params"]["tuned_on_users"] = int(len(tune_idx))
    results["tuning"] = tuning
    results["tuning_std"] = tuning_std
    (models_dir / "bandit_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    for name, p in results["policies"].items():
        _log(f"{name}: mean reward {p['mean_reward']:.4f}, cum regret {p['cum_regret']:.2f} "
             f"+/- {p['cum_regret_std']:.2f}, arm counts {p['arm_counts']}")
    return _summary(results)


# ----------------------------------------------------------------------------- evaluation
POLICY_LABELS = {"random": "Random", "epsilon_greedy": "Epsilon-greedy", "linucb": "LinUCB"}


def _curve_x(n_points: int, rounds: int) -> np.ndarray:
    """Round numbers (1-based) of the subsampled curve points (matches `_curve_idx`)."""
    return np.minimum(CURVE_STEP * (np.arange(n_points) + 1), rounds)


def plot_results(results: dict, figures_dir: Path) -> None:
    """bandit_cumulative_reward.png (with the oracle's expected line) and bandit_regret.png (± std)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    rounds, n_seeds = results["rounds"], results["n_seeds"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, p in results["policies"].items():
        y = np.asarray(p["cum_reward_curve"])
        ax.plot(_curve_x(len(y), rounds), y, label=POLICY_LABELS.get(name, name))
    ax.plot([0, rounds], [0, rounds * results["oracle_mean_reward"]], "k--", lw=1, label="Oracle (expected)")
    ax.set_xlabel("round")
    ax.set_ylabel("cumulative reward (fraction of 10 items clicked)")
    ax.set_title(f"Bandit cumulative reward (mean of {n_seeds} seeds)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "bandit_cumulative_reward.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, p in results["policies"].items():
        y, s = np.asarray(p["cum_regret_curve"]), np.asarray(p["cum_regret_curve_std"])
        x = _curve_x(len(y), rounds)
        ax.plot(x, y, label=POLICY_LABELS.get(name, name))
        ax.fill_between(x, y - s, y + s, alpha=0.2)
    ax.set_xlabel("round")
    ax.set_ylabel("cumulative expected regret")
    ax.set_title(f"Bandit cumulative regret (mean ± std over {n_seeds} seeds)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "bandit_regret.png", dpi=150)
    plt.close(fig)


def eval_step(cfg, ctx: dict) -> dict | None:
    """Figures from models/bandit_results.json; returns the metrics "bandit" dict (None if missing)."""
    models_dir, _, figures_dir = _dirs(ctx)
    path = models_dir / "bandit_results.json"
    if not path.exists():
        _log(f"{path} missing; skipping")
        return None
    results = json.loads(path.read_text(encoding="utf-8"))
    plot_results(results, figures_dir)
    return _summary(results)
