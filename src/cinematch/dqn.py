"""DQN stretch goal: learn the ORDER of a 5-slot recommendation session from 8 CF candidates.

Candidates for a user are the first 8 items of the bandit simulator's "cf" list
(`models/bandit_env.npz`), each with its true click probability p_j from the test-split click model.
Session dynamics make the order matter: at step t (0..4) showing candidate j is clicked with

    p_j * position_decay**t * fatigue**overlap_j

where overlap_j = number of already-shown items sharing >= 1 genre with j (repeats are masked).
Early slots are worth more, and showing similar items back to back tires the user. Because the
state transitions do not depend on clicks, the expected session reward of any order is exact, so
every policy (DQN, CF order, random orders, a diversity heuristic, and the brute-force oracle over
all 8P5 = 6,720 orders) is evaluated analytically, without sampling noise.

Only users with some click signal (a candidate with p_j > click_unrated) are used; they are split
50/50 (seeded) into DQN train and eval users. The DQN never sees p_j: its state is
[t/5, used_mask(8), shown_genre_counts(18)/5, candidate_genres(8x18)], so it can only exploit the
slot position (= CF rank) and genre structure. Training uses Bernoulli clicks.

Torch is imported lazily inside functions: importing this module loads numpy only.
"""
from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

import numpy as np

from cinematch.config import FIGURES_DIR, MODELS_DIR, PROCESSED_DIR

N_CAND = 8                       # candidate items per user (top-8 of the CF arm)
SESSION_LEN = 5                  # slots per session
N_GENRES = 18
STATE_DIM = 1 + N_CAND + N_GENRES + N_CAND * N_GENRES   # 171
POLICIES = ["dqn", "greedy_cf", "random", "diversity_heuristic"]
N_RANDOM_ORDERS = 20
CURVE_WINDOW = 100               # moving-average window of the training curve
CURVE_STEP = 10                  # curves are subsampled every CURVE_STEP episodes in the json


def _log(msg: str) -> None:
    print(f"[dqn] {msg}")


def _torch():
    import torch
    return torch


# ----------------------------------------------------------------------------- session math
def genre_overlap(genres: np.ndarray) -> np.ndarray:
    """[n, 8, 18] candidate genres -> [n, 8, 8] bool: candidates j and k share >= 1 genre."""
    g = np.asarray(genres, dtype=np.float32)
    return np.einsum("nig,njg->nij", g, g) > 0


def step_click_probs(p: np.ndarray, share: np.ndarray, used: np.ndarray, t: int,
                     decay: float, fatigue: float) -> np.ndarray:
    """Click probability of every candidate if it were shown now (step t, `used` already shown).

    Shapes: p [n, 8], share [n, 8, 8] bool, used [n, 8] bool -> [n, 8].
    """
    overlap = (share & used[:, None, :]).sum(axis=-1)
    return p * decay ** t * fatigue ** overlap


def order_expected_reward(p: np.ndarray, share: np.ndarray, orders: np.ndarray,
                          decay: float, fatigue: float) -> np.ndarray:
    """Exact expected clicks of a session shown in the given order.

    `orders` is [n, L] (one order per user) or [n, P, L] (P orders per user) of distinct slot
    indices; returns [n] or [n, P]. Implements sum_t p[a_t] decay^t fatigue^(sum_{s<t} share[a_t, a_s]).
    """
    orders = np.asarray(orders, dtype=np.int64)
    squeeze = orders.ndim == 2
    if squeeze:
        orders = orders[:, None, :]
    rows = np.arange(len(orders))[:, None]
    total = np.zeros(orders.shape[:2])
    for t in range(orders.shape[2]):
        a_t = orders[:, :, t]
        overlap = np.zeros(a_t.shape)
        for s in range(t):
            overlap += share[rows, a_t, orders[:, :, s]]
        total += p[rows, a_t] * decay ** t * fatigue ** overlap
    return total[:, 0] if squeeze else total


def build_states(t: int, used: np.ndarray, genres: np.ndarray) -> np.ndarray:
    """DQN states [n, STATE_DIM]: [t/5, used_mask(8), shown_genre_counts(18)/5, candidate_genres(144)]."""
    used = np.asarray(used, dtype=np.float32)
    genres = np.asarray(genres, dtype=np.float32)
    n = len(used)
    shown = (used[:, :, None] * genres).sum(axis=1) / SESSION_LEN
    return np.hstack([np.full((n, 1), t / SESSION_LEN, np.float32), used, shown,
                      genres.reshape(n, -1)]).astype(np.float32)


# ----------------------------------------------------------------------------- data
def build_session_data(env: dict, genre_multi_hot: np.ndarray, click_unrated: float) -> dict:
    """Candidates, click probs and genres for every bandit user, plus the has-signal mask.

    Returns {"user_idx" [n], "items" [n, 8], "p" [n, 8], "genres" [n, 8, 18], "share" [n, 8, 8],
    "valid" [n, 8] bool, "signal" [n] bool}. A padded slot (item -1) gets p = 0, no genres and
    valid = False; every policy shows it only after all real candidates.
    """
    arm = list(np.asarray(env["arm_names"]).astype(str)).index("cf")
    items = np.asarray(env["lists"])[:, arm, :N_CAND].astype(np.int64)
    p = np.asarray(env["probs"])[:, arm, :N_CAND].astype(np.float64)
    gmh = np.asarray(genre_multi_hot, dtype=np.float32)
    genres = gmh[np.clip(items, 0, None)]
    genres[items < 0] = 0.0
    p[items < 0] = 0.0
    signal = (p > float(np.float32(click_unrated)) + 1e-6).any(axis=1)
    return {"user_idx": np.asarray(env["user_idx"]), "items": items, "p": p, "genres": genres,
            "share": genre_overlap(genres), "valid": items >= 0, "signal": signal}


def _valid(data: dict) -> np.ndarray:
    """[n, 8] bool real-candidate mask (all True when `data` carries no "valid" entry)."""
    v = data.get("valid")
    return np.ones(np.shape(data["p"]), bool) if v is None else np.asarray(v, bool)


def _blocked(used: np.ndarray, valid: np.ndarray | None) -> np.ndarray:
    """Slots a policy may not pick now: already shown, or padded while a real candidate is left."""
    if valid is None:
        return used
    blocked = used | ~valid
    exhausted = blocked.all(axis=-1, keepdims=True)
    return np.where(exhausted, used, blocked)


def subset(data: dict, rows: np.ndarray) -> dict:
    """Restrict every per-user array of `data` to `rows`."""
    return {k: np.asarray(v)[rows] for k, v in data.items()}


def split_users(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Seeded 50/50 split of row indices 0..n-1 into (train, eval), each sorted."""
    perm = np.random.default_rng(seed).permutation(n)
    half = n // 2
    return np.sort(perm[:half]), np.sort(perm[half:])


# ----------------------------------------------------------------------------- baseline policies
def greedy_cf_orders(n: int) -> np.ndarray:
    """Slots in CF rank order 0..4 for every user."""
    return np.tile(np.arange(SESSION_LEN), (n, 1))


def random_orders(n: int, seed: int, valid: np.ndarray | None = None) -> np.ndarray:
    """One uniformly random order of 5 distinct slots per user (padded slots, if any, last)."""
    rng = np.random.default_rng(seed)
    key = rng.random((n, N_CAND))
    if valid is not None:
        key = key + (~np.asarray(valid, bool))   # keys are < 1, so invalid slots sort after valid ones
    return np.argsort(key, axis=1, kind="stable")[:, :SESSION_LEN]


def diversity_orders(share: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    """At each step pick the unused slot with the smallest genre overlap; ties -> best CF rank.

    Padded slots (valid False) are picked only once no real candidate is left.
    """
    n = len(share)
    used = np.zeros((n, N_CAND), dtype=bool)
    orders = np.zeros((n, SESSION_LEN), dtype=np.int64)
    rows = np.arange(n)
    for t in range(SESSION_LEN):
        overlap = (share & used[:, None, :]).sum(axis=-1).astype(np.float64)
        key = overlap * (N_CAND + 1) + np.arange(N_CAND)[None, :]
        key[_blocked(used, valid)] = np.inf
        a = key.argmin(axis=1)
        orders[:, t] = a
        used[rows, a] = True
    return orders


def all_orders() -> np.ndarray:
    """All 8P5 = 6,720 ordered selections of 5 distinct slots, [6720, 5]."""
    return np.array(list(itertools.permutations(range(N_CAND), SESSION_LEN)), dtype=np.int64)


def oracle_reward(p: np.ndarray, share: np.ndarray, decay: float, fatigue: float) -> np.ndarray:
    """Best achievable expected session reward per user (brute force over all orders), [n]."""
    perms = all_orders()
    n = len(p)
    orders = np.broadcast_to(perms[None], (n,) + perms.shape)
    return order_expected_reward(p, share, orders, decay, fatigue).max(axis=1)


def baseline_rewards(data: dict, cfg) -> dict:
    """Mean expected session reward of greedy_cf, random (20 seeded orders), diversity and oracle."""
    p, share, valid = data["p"], data["share"], _valid(data)
    d, f, n = cfg.dqn_position_decay, cfg.dqn_fatigue, len(p)
    rnd = np.mean([order_expected_reward(p, share, random_orders(n, cfg.seed + s, valid), d, f).mean()
                   for s in range(N_RANDOM_ORDERS)])
    div = order_expected_reward(p, share, diversity_orders(share, valid), d, f).mean()
    return {"greedy_cf": float(order_expected_reward(p, share, greedy_cf_orders(n), d, f).mean()),
            "random": float(rnd),
            "diversity_heuristic": float(div),
            "oracle": float(oracle_reward(p, share, d, f).mean())}


# ----------------------------------------------------------------------------- Q-network
def build_qnet(in_dim: int = STATE_DIM, hidden: int = 128, n_actions: int = N_CAND):
    """MLP state -> hidden -> hidden -> Q-value per candidate slot (torch, built lazily)."""
    nn = _torch().nn
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                         nn.Linear(hidden, n_actions))


def export_weights(net) -> dict[str, np.ndarray]:
    """Linear layers of a `build_qnet` model as numpy arrays W0, b0, W1, b1, W2, b2."""
    layers = [m for m in net if hasattr(m, "weight")]
    out = {}
    for k, m in enumerate(layers):
        out[f"W{k}"] = m.weight.detach().cpu().numpy().astype(np.float32)
        out[f"b{k}"] = m.bias.detach().cpu().numpy().astype(np.float32)
    return out


def q_values_np(weights: dict, states: np.ndarray) -> np.ndarray:
    """Numpy forward pass of the exported Q-network (no torch needed), [n, 8]."""
    h = np.asarray(states, dtype=np.float32)
    n_layers = sum(1 for k in weights if str(k).startswith("W"))
    for k in range(n_layers):
        h = h @ weights[f"W{k}"].T + weights[f"b{k}"]
        if k < n_layers - 1:
            h = np.maximum(h, 0.0)
    return h


def policy_orders(q_fn, genres: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    """Greedy rollout of a Q-function: at each step the unused slot with the highest Q-value.

    `q_fn(states [n, STATE_DIM]) -> [n, 8]`. Transitions are deterministic, so one rollout per user
    gives the policy's order. Padded slots (valid False) are masked while a real candidate is left.
    """
    n = len(genres)
    used = np.zeros((n, N_CAND), dtype=bool)
    orders = np.zeros((n, SESSION_LEN), dtype=np.int64)
    rows = np.arange(n)
    for t in range(SESSION_LEN):
        q = np.array(q_fn(build_states(t, used, genres)), dtype=np.float64)
        q[_blocked(used, valid)] = -np.inf
        a = q.argmax(axis=1)
        orders[:, t] = a
        used[rows, a] = True
    return orders


def _torch_q_fn(net):
    torch = _torch()

    def q_fn(states):
        with torch.no_grad():
            return net(torch.from_numpy(states)).numpy()
    return q_fn


def dqn_reward(q_fn, data: dict, cfg) -> float:
    """Mean exact expected session reward of the greedy policy of `q_fn` on `data`'s users."""
    orders = policy_orders(q_fn, data["genres"], _valid(data))
    return float(order_expected_reward(data["p"], data["share"], orders,
                                       cfg.dqn_position_decay, cfg.dqn_fatigue).mean())


# ----------------------------------------------------------------------------- training
class ReplayBuffer:
    """Fixed-size ring buffer of (state, action, reward, next_state, next_valid_mask, done)."""

    def __init__(self, capacity: int):
        self.capacity, self.size, self.pos = int(capacity), 0, 0
        self.s = np.zeros((self.capacity, STATE_DIM), np.float32)
        self.s2 = np.zeros((self.capacity, STATE_DIM), np.float32)
        self.a = np.zeros(self.capacity, np.int64)
        self.r = np.zeros(self.capacity, np.float32)
        self.valid2 = np.zeros((self.capacity, N_CAND), bool)
        self.done = np.zeros(self.capacity, np.float32)

    def add(self, s, a, r, s2, valid2, done) -> None:
        i = self.pos
        self.s[i], self.a[i], self.r[i], self.s2[i], self.valid2[i], self.done[i] = s, a, r, s2, valid2, done
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch: int, rng: np.random.Generator) -> tuple:
        idx = rng.integers(self.size, size=batch)
        return self.s[idx], self.a[idx], self.r[idx], self.s2[idx], self.valid2[idx], self.done[idx]


def epsilon_at(episode: int, cfg) -> float:
    """Linear decay 1.0 -> cfg.dqn_eps_end over the first cfg.dqn_eps_decay_frac of the episodes."""
    span = max(1.0, cfg.dqn_eps_decay_frac * cfg.dqn_episodes)
    return float(max(cfg.dqn_eps_end, 1.0 - (1.0 - cfg.dqn_eps_end) * episode / span))


def train_dqn(train: dict, cfg, eval_data: dict | None = None, log=_log) -> tuple:
    """Train the session-ordering DQN on `train` users with Bernoulli clicks.

    Vanilla DQN (no double-Q): replay buffer, target network synced every cfg.dqn_target_sync
    gradient steps, epsilon-greedy over unused slots, masked max in the bootstrap target, Huber loss.
    Returns (net, history) where history holds per-episode Bernoulli and expected rewards and a
    periodic greedy evaluation on the train (and optional eval) users.
    """
    torch = _torch()
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    d, f, L = cfg.dqn_position_decay, cfg.dqn_fatigue, SESSION_LEN
    net = build_qnet(STATE_DIM, cfg.dqn_hidden, N_CAND)
    target = build_qnet(STATE_DIM, cfg.dqn_hidden, N_CAND)
    target.load_state_dict(net.state_dict())
    opt = torch.optim.Adam(net.parameters(), lr=cfg.dqn_lr)
    buf = ReplayBuffer(cfg.dqn_buffer)
    p, share, genres, valid = train["p"], train["share"], train["genres"], _valid(train)
    n = len(p)
    q_fn = _torch_q_fn(net)
    hist = {"episode_reward": [], "episode_expected": [], "eval_episode": [], "eval_train": [], "eval_eval": []}
    grad_steps = 0

    def greedy_eval(ep):
        hist["eval_episode"].append(ep)
        hist["eval_train"].append(dqn_reward(q_fn, train, cfg))
        hist["eval_eval"].append(dqn_reward(q_fn, eval_data, cfg) if eval_data is not None else None)

    for ep in range(cfg.dqn_episodes):
        if ep % cfg.dqn_eval_every == 0:
            greedy_eval(ep)
        eps = epsilon_at(ep, cfg)
        u = int(rng.integers(n))
        used = np.zeros(N_CAND, dtype=bool)
        s = build_states(0, used[None], genres[u][None])[0]
        total = expected = 0.0
        for t in range(L):
            blocked = _blocked(used, valid[u])
            if rng.random() < eps:
                a = int(rng.choice(np.flatnonzero(~blocked)))
            else:
                q = q_fn(s[None])[0].astype(np.float64)
                q[blocked] = -np.inf
                a = int(q.argmax())
            prob = float(step_click_probs(p[u][None], share[u][None], used[None], t, d, f)[0, a])
            r = float(rng.random() < prob)
            total, expected = total + r, expected + prob
            used = used.copy()
            used[a] = True
            s2 = build_states(t + 1, used[None], genres[u][None])[0]
            buf.add(s, a, r, s2, ~_blocked(used, valid[u]), float(t == L - 1))
            s = s2
            if buf.size >= max(cfg.dqn_batch, cfg.dqn_warmup):
                bs, ba, br, bs2, bv2, bd = (torch.from_numpy(x) for x in buf.sample(cfg.dqn_batch, rng))
                q_sa = net(bs).gather(1, ba[:, None]).squeeze(1)
                with torch.no_grad():
                    q_next = target(bs2).masked_fill(~bv2, -1e9).max(dim=1).values
                    y = br + cfg.dqn_gamma * (1.0 - bd) * q_next
                loss = torch.nn.functional.smooth_l1_loss(q_sa, y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                grad_steps += 1
                if grad_steps % cfg.dqn_target_sync == 0:
                    target.load_state_dict(net.state_dict())
        hist["episode_reward"].append(total)
        hist["episode_expected"].append(float(expected))
    greedy_eval(cfg.dqn_episodes)
    hist["grad_steps"] = grad_steps
    log(f"trained {cfg.dqn_episodes} episodes ({grad_steps} gradient steps); greedy expected reward "
        f"train {hist['eval_train'][-1]:.4f}" + (f", eval {hist['eval_eval'][-1]:.4f}" if eval_data else ""))
    return net, hist


# ----------------------------------------------------------------------------- pipeline steps
def _dirs(ctx: dict) -> tuple[Path, Path, Path]:
    return (Path(ctx.get("models_dir", MODELS_DIR)), Path(ctx.get("processed_dir", PROCESSED_DIR)),
            Path(ctx.get("figures_dir", FIGURES_DIR)))


def _load_genres(ctx: dict, processed_dir: Path) -> np.ndarray | None:
    """genre_multi_hot [n_items, 18] from ctx["data"] if loaded, else processed item_features.npz."""
    feats = (ctx.get("data") or {}).get("item_features")
    if feats is not None and "genre_multi_hot" in feats:
        return np.asarray(feats["genre_multi_hot"])
    path = processed_dir / "item_features.npz"
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as z:
        return z["genre_multi_hot"]


def moving_average(x, window: int = CURVE_WINDOW) -> tuple[np.ndarray, np.ndarray]:
    """(episode numbers, trailing moving average) of a per-episode series (window capped at len)."""
    x = np.asarray(x, dtype=np.float64)
    w = max(1, min(int(window), len(x)))
    ma = np.convolve(x, np.ones(w) / w, mode="valid")
    return np.arange(w, len(x) + 1), ma


def _r4(d: dict) -> dict:
    return {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v) for k, v in d.items()}


def train_step(cfg, ctx: dict) -> dict | None:
    """Train the session-ordering DQN on bandit_env users; write models/dqn.npz + dqn_results.json.

    Needs models/bandit_env.npz and processed item_features.npz (returns None, writing nothing,
    if either is missing or fewer than 2 users have click signal). Returns the metrics "dqn" dict.
    """
    models_dir, processed_dir, _ = _dirs(ctx)
    t0 = time.time()
    env_path = models_dir / "bandit_env.npz"
    genres = _load_genres(ctx, processed_dir)
    if not env_path.exists() or genres is None:
        _log(f"{env_path} or item_features.npz missing; skipping")
        return None
    from cinematch.bandit import load_env
    data = build_session_data(load_env(models_dir), genres, cfg.click_unrated)
    sig = np.flatnonzero(data["signal"])
    _log(f"{len(sig)} of {len(data['p'])} bandit users have click signal in their top-{N_CAND} CF candidates")
    if len(sig) < 2:
        _log("too few users with signal; skipping")
        return None
    tr, ev = split_users(len(sig), cfg.seed)
    train, evald = subset(data, sig[tr]), subset(data, sig[ev])

    net, hist = train_dqn(train, cfg, evald)
    weights = export_weights(net)
    q_np = lambda s: q_values_np(weights, s)  # noqa: E731 - numpy policy = exactly what is saved
    base_eval, base_train = baseline_rewards(evald, cfg), baseline_rewards(train, cfg)
    metrics = _r4({"dqn": dqn_reward(q_np, evald, cfg), "greedy_cf": base_eval["greedy_cf"],
                   "random": base_eval["random"], "diversity_heuristic": base_eval["diversity_heuristic"],
                   "oracle": base_eval["oracle"], "n_train_users": len(tr), "n_eval_users": len(ev),
                   "n_users_total": int(len(data["p"])), "n_users_with_signal": int(len(sig)),
                   "episodes": int(cfg.dqn_episodes), "session_len": SESSION_LEN})
    train_users = _r4({"dqn": dqn_reward(q_np, train, cfg), **base_train})

    models_dir.mkdir(parents=True, exist_ok=True)
    np.savez(models_dir / "dqn.npz", **weights, train_user_idx=train["user_idx"], eval_user_idx=evald["user_idx"],
             position_decay=np.float32(cfg.dqn_position_decay), fatigue=np.float32(cfg.dqn_fatigue))
    x, ma_r = moving_average(hist["episode_reward"])
    _, ma_e = moving_average(hist["episode_expected"])
    keep = np.unique(np.r_[np.arange(0, len(x), CURVE_STEP), len(x) - 1])
    results = {
        "metrics": metrics,
        "train_users": train_users,
        "n_users_total": int(len(data["p"])), "n_users_with_signal": int(len(sig)),
        "params": {"position_decay": cfg.dqn_position_decay, "fatigue": cfg.dqn_fatigue, "gamma": cfg.dqn_gamma,
                   "lr": cfg.dqn_lr, "hidden": cfg.dqn_hidden, "batch": cfg.dqn_batch, "buffer": cfg.dqn_buffer,
                   "target_sync": cfg.dqn_target_sync, "eps_end": cfg.dqn_eps_end,
                   "eps_decay_frac": cfg.dqn_eps_decay_frac, "grad_steps": hist["grad_steps"]},
        "curve": {"window": CURVE_WINDOW, "episode": x[keep].tolist(),
                  "reward_ma": np.round(ma_r[keep], 4).tolist(), "expected_ma": np.round(ma_e[keep], 4).tolist()},
        "eval_curve": {"episode": hist["eval_episode"], "train_users": np.round(hist["eval_train"], 4).tolist(),
                       "eval_users": np.round(hist["eval_eval"], 4).tolist()},
        "train_seconds": round(time.time() - t0, 1),
    }
    (models_dir / "dqn_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    _log(f"eval users ({len(ev)}): " + ", ".join(f"{k} {metrics[k]:.4f}" for k in POLICIES + ["oracle"])
         + f"; {results['train_seconds']}s")
    return metrics


# ----------------------------------------------------------------------------- evaluation
def plot_training(results: dict, path: Path) -> None:
    """dqn_training.png: moving-average training reward, greedy-policy checkpoints, baseline lines."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curve, ev, base = results["curve"], results["eval_curve"], results["train_users"]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(curve["episode"], curve["reward_ma"], color="tab:blue", alpha=0.35, lw=1,
            label=f"training clicks (Bernoulli, MA {curve['window']})")
    ax.plot(curve["episode"], curve["expected_ma"], color="tab:blue", lw=1.8,
            label=f"training expected reward (MA {curve['window']})")
    ax.plot(ev["episode"], ev["train_users"], "o-", color="tab:green", ms=3, lw=1, label="greedy DQN, train users")
    if ev.get("eval_users") and ev["eval_users"][0] is not None:
        ax.plot(ev["episode"], ev["eval_users"], "s-", color="tab:red", ms=3, lw=1, label="greedy DQN, eval users")
    styles = {"greedy_cf": ("gray", "--"), "diversity_heuristic": ("tab:purple", "--"),
              "random": ("black", ":"), "oracle": ("goldenrod", "-.")}
    for name, (color, ls) in styles.items():
        if name in base:
            ax.axhline(base[name], color=color, ls=ls, lw=1, label=f"{name} (train users)")
    ax.axhline(results["metrics"]["greedy_cf"], color="tab:red", ls="--", lw=0.8, alpha=0.6,
               label="greedy_cf (eval users)")
    ax.set_xlabel("episode")
    ax.set_ylabel(f"clicks per {SESSION_LEN}-slot session")
    ax.set_title("DQN session ordering: training (epsilon-greedy) vs baselines")
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def eval_step(cfg, ctx: dict) -> dict | None:
    """Figure dqn_training.png from models/dqn_results.json; returns the metrics "dqn" dict.

    Returns None (no error) if models/dqn.npz or models/dqn_results.json is missing.
    """
    models_dir, _, figures_dir = _dirs(ctx)
    res_path = models_dir / "dqn_results.json"
    if not res_path.exists() or not (models_dir / "dqn.npz").exists():
        _log(f"{res_path} or dqn.npz missing; skipping")
        return None
    results = json.loads(res_path.read_text(encoding="utf-8"))
    plot_training(results, figures_dir / "dqn_training.png")
    return results["metrics"]
