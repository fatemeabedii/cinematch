"""CineMatch Streamlit app: recommendations, taste profiles, a live bandit lab and an explainer.

Inference is numpy-only (no torch at import time): models are loaded from the `.npz`/`.json`
artifacts written by `scripts/train_all.py`; every number shown on the "How It Works" tab is
read from `reports/metrics.json` (written by `scripts/evaluate.py`) at runtime.

Run locally:  streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import html  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import dataclasses  # noqa: E402
import re  # noqa: E402
from urllib.parse import quote  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import requests  # noqa: E402
import streamlit as st  # noqa: E402

from cinematch import anomaly  # noqa: E402
from cinematch.config import (  # noqa: E402
    FIGURES_DIR, MODELS_DIR, PROCESSED_DIR, RELEVANT_THRESHOLD, REPORTS_DIR, get_config,
)

try:  # recommend.py / bandit.py are numpy-only; guard so a partial checkout still renders a message
    from cinematch.recommend import MODEL_FILES, PROCESSED_FILES, ModelsMissingError, Recommender
except ImportError:  # pragma: no cover - only when the package is incomplete
    Recommender = None
    MODEL_FILES, PROCESSED_FILES = ("cf_mf.npz",), ()

    class ModelsMissingError(FileNotFoundError):
        """Placeholder used when cinematch.recommend is not available."""
try:
    from cinematch import bandit
except ImportError:  # pragma: no cover
    bandit = None

REPO_URL = "https://github.com/<user>/cinematch"
RUN_INSTRUCTIONS = (
    "python scripts/download_data.py\n"
    "python scripts/train_all.py\n"
    "python scripts/evaluate.py"
)
METRICS_PATH = REPORTS_DIR / "metrics.json"
STRATEGIES = {"Collaborative filtering (MF)": "cf", "Content (two-tower)": "content", "Hybrid": "hybrid"}
# Display label per bandit policy key (bandit.POLICY_NAMES); also used for the metrics.json table.
POLICY_LABELS = getattr(bandit, "POLICY_LABELS", {"random": "Random", "epsilon_greedy": "Epsilon-greedy",
                                                  "linucb": "LinUCB"})
# What the card score means per strategy (recommend.py: cf/content = predicted rating clipped to 1..5,
# hybrid = mean of per-model z-scores (only meaningful relative to the other movies in the list),
# cluster = profile score with a popularity bonus, not a rating).
SCORE_LABELS = {"cf": "predicted rating", "content": "predicted rating", "hybrid": "relative score",
                "cluster": "profile score"}
# Alpha values offered in the Bandit Lab (log-like spacing; the tuned value is always added).
ALPHA_OPTIONS = (0.0, 0.01, 0.03, 0.1, 0.3, 0.5, 1.0, 2.0, 3.0)
MODEL_LABELS = {
    "popularity": "Popularity",
    "bias_baseline": "Bias baseline",
    "cf_mf": "Matrix factorization (CF)",
    "two_tower": "Two-tower content model",
}


# ----------------------------------------------------------------------------- cached loaders
# Every cached loader takes a `stamp` (paths + modification times of the files it reads) as its
# cache key, so retraining while the server runs, or pointing MODELS_DIR elsewhere, is picked up
# on the next rerun instead of serving a stale (or stale "missing") result until restart.
def _stamp(*paths: Path) -> tuple:
    """(path, mtime or None) for each file: a cache key that changes when a file is (re)written."""
    return tuple((str(p), p.stat().st_mtime if p.exists() else None) for p in paths)


def _recommender_stamp() -> tuple:
    """Stamp of every artifact the Recommender reads (movies.pkl is recommend.py's parquet fallback)."""
    files = [MODELS_DIR / f for f in MODEL_FILES] + [PROCESSED_DIR / f for f in (*PROCESSED_FILES, "movies.pkl")]
    return _stamp(*files)


@st.cache_resource(show_spinner="Loading models...", max_entries=1)
def get_recommender(stamp: tuple):
    """The numpy Recommender, cached per artifact stamp. Raises ModelsMissingError (never cached)."""
    if Recommender is None:
        raise ModelsMissingError("cinematch.recommend is not available")
    return Recommender(MODELS_DIR, PROCESSED_DIR)


def load_recommender():
    """Recommender, or None after showing a friendly warning with run instructions."""
    try:
        return get_recommender(_recommender_stamp())
    except ModelsMissingError as err:
        st.warning("Models not found. Train them first (from the repo root):")
        st.code(RUN_INSTRUCTIONS, language="bash")
        st.caption(f"Details: {err}")
        return None


@st.cache_data(show_spinner=False, max_entries=4)
def _load_metrics(stamp: tuple) -> dict | None:
    try:
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def load_metrics() -> dict | None:
    """reports/metrics.json as a dict, or None if evaluate.py has not been run."""
    return _load_metrics(_stamp(METRICS_PATH))


@st.cache_data(show_spinner=False, max_entries=4)
def _popular_movies(n: int, stamp: tuple) -> pd.DataFrame:
    return get_recommender(stamp).popular_for_rating(n)


def popular_movies(n: int = 30) -> pd.DataFrame:
    return _popular_movies(n, _recommender_stamp())


@st.cache_data(show_spinner=False, max_entries=4)
def _bandit_env(stamp: tuple) -> dict | None:
    if bandit is None or not (MODELS_DIR / "bandit_env.npz").exists():
        return None
    try:
        return bandit.load_env(MODELS_DIR)
    except (OSError, ValueError):
        return None


def bandit_env() -> dict | None:
    """Precomputed bandit simulator (models/bandit_env.npz), or None if it was not built."""
    return _bandit_env(_stamp(MODELS_DIR / "bandit_env.npz"))


@st.cache_data(show_spinner=False, max_entries=4)
def _bandit_params(stamp: tuple) -> dict:
    try:
        params = json.loads((MODELS_DIR / "bandit_results.json").read_text(encoding="utf-8")).get("params")
    except (OSError, ValueError, AttributeError):
        return {}
    return params if isinstance(params, dict) else {}


def bandit_defaults() -> dict:
    """Bandit Lab defaults: the tuned `params` of models/bandit_results.json, else the config values.

    Returns {"epsilon", "alpha", "tuned_on_users" (int or None), "source": "tuned" | "results" | "config"}.
    """
    cfg = get_config()
    params = _bandit_params(_stamp(MODELS_DIR / "bandit_results.json"))
    out = {"epsilon": float(cfg.bandit_epsilon), "alpha": float(cfg.bandit_alpha), "tuned_on_users": None,
           "source": "config"}
    try:
        if params.get("epsilon") is not None and params.get("alpha") is not None:
            out.update(epsilon=float(params["epsilon"]), alpha=float(params["alpha"]), source="results")
            if params.get("tuned_on_users"):
                out.update(tuned_on_users=int(params["tuned_on_users"]), source="tuned")
    except (TypeError, ValueError):
        pass
    return out


@st.cache_data(show_spinner=False, max_entries=4)
def _anomaly_model(stamp: tuple) -> dict | None:
    try:
        return anomaly.load_anomaly(MODELS_DIR)
    except (OSError, ValueError, KeyError):
        return None


def anomaly_model() -> dict | None:
    """The saved Gaussian bot detector (models/anomaly.npz), or None if it is missing or unreadable."""
    return _anomaly_model(_stamp(MODELS_DIR / "anomaly.npz"))


# ----------------------------------------------------------------------------- posters
def _search_title(title: str) -> str:
    """'Matrix, The (1999)' -> 'The Matrix' (TMDB search friendly)."""
    t = re.sub(r"\s*\(\d{4}\)\s*$", "", str(title)).strip()
    m = re.match(r"^(.*), (The|A|An|Les|La|Le|Il|El|Das|Der|Die)$", t)
    return f"{m.group(2)} {m.group(1)}" if m else t


def placeholder_poster(title: str) -> str:
    """Offline SVG poster (data URI) with the movie title."""
    words, lines, line = _search_title(title).split(), [], ""
    for w in words:
        if len(line) + len(w) > 16 and line:
            lines.append(line)
            line = w
        else:
            line = f"{line} {w}".strip()
    lines.append(line)
    tspans = "".join(
        f'<tspan x="150" dy="{0 if j == 0 else 34}">{html.escape(s)}</tspan>' for j, s in enumerate(lines[:5])
    )
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="300" height="450">'
        '<defs><linearGradient id="g" x1="0" y1="0" x2="0" y2="1">'
        '<stop offset="0" stop-color="#2b2d42"/><stop offset="1" stop-color="#141414"/></linearGradient></defs>'
        '<rect width="300" height="450" fill="url(#g)"/>'
        f'<text x="150" y="{200 - 17 * (len(lines[:5]) - 1)}" fill="#f5f5f5" font-family="Helvetica,Arial" '
        f'font-size="26" text-anchor="middle">{tspans}</text>'
        '<text x="150" y="420" fill="#E50914" font-family="Helvetica,Arial" font-size="18" '
        'text-anchor="middle">CineMatch</text></svg>'
    )
    return "data:image/svg+xml;charset=utf-8," + quote(svg)


@st.cache_data(show_spinner=False)
def get_poster_url(title: str, year: int | None = None) -> str | None:
    """TMDB poster URL if TMDB_API_KEY is set and the lookup succeeds, else None."""
    key = os.environ.get("TMDB_API_KEY")
    if not key:
        return None
    try:
        params = {"api_key": key, "query": _search_title(title)}
        if year:
            params["year"] = int(year)
        r = requests.get("https://api.themoviedb.org/3/search/movie", params=params, timeout=3)
        results = r.json().get("results") or []
        path = results[0].get("poster_path") if results else None
        return f"https://image.tmdb.org/t/p/w342{path}" if path else None
    except Exception:  # noqa: BLE001 - posters are decoration; never break the app
        return None


# ----------------------------------------------------------------------------- small helpers
def _fmt(value, digits: int = 4) -> str:
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def _genres_pp(top_genres, n: int = 3) -> str:
    """[[genre, pp], ...] -> 'Drama +4.1 pp, Film-Noir +2.0 pp' (over-representation vs all users)."""
    return ", ".join(f"{g} {float(v):+.1f} pp" for g, v in list(top_genres)[:n])


def movie_cards(df: pd.DataFrame, score_label: str | None = "predicted rating", per_row: int = 5) -> None:
    """Render recommendations as poster cards (title, genres, score, why).

    `score_label` names what the score means for this strategy (scales differ); None hides it.
    """
    if df is None or len(df) == 0:
        st.info("No recommendations available.")
        return
    rows = df.to_dict("records")
    for start in range(0, len(rows), per_row):
        cols = st.columns(per_row)
        for col, row in zip(cols, rows[start:start + per_row]):
            title = str(row.get("title", ""))
            poster = get_poster_url(title, row.get("year")) or placeholder_poster(title)
            genres = str(row.get("genres", "")).replace("|", " · ")
            score = row.get("score")
            show = score_label and score is not None and np.isfinite(score)
            score_txt = f"{score_label} {float(score):.2f}" if show else ""
            why = html.escape(str(row.get("why", "") or ""))
            col.markdown(
                f'<img src="{html.escape(poster, quote=True)}" alt="" style="width:100%;aspect-ratio:2/3;object-fit:cover;'
                f'border-radius:8px;">'
                f'<div style="font-weight:600;margin-top:6px;line-height:1.25">{html.escape(title)}</div>'
                f'<div style="font-size:0.8em;opacity:0.75">{html.escape(genres)}</div>'
                f'<div style="font-size:0.8em;opacity:0.75">{score_txt}</div>'
                f'<div style="font-size:0.8em;font-style:italic;opacity:0.85">{why}</div>',
                unsafe_allow_html=True,
            )


# ----------------------------------------------------------------------------- tab 1
def _rating_form(rec) -> None:
    movies = popular_movies(30)
    st.caption("Rate at least 3 movies you know (leave the rest on *skip*), pick a model, then hit Recommend.")
    with st.form("rate_form"):
        cols = st.columns(3)
        for j, row in enumerate(movies.to_dict("records")):
            cols[j % 3].select_slider(
                str(row["title"]), options=["skip", "1", "2", "3", "4", "5"], value="skip",
                key=f"rate_{int(row['movie_id'])}",
            )
        strategy_label = st.radio("Model", list(STRATEGIES), horizontal=True, key="strategy")
        submitted = st.form_submit_button("Recommend", type="primary")

    if submitted:
        ratings = {}
        for mid in movies["movie_id"]:
            value = st.session_state.get(f"rate_{int(mid)}", "skip")
            if value != "skip":
                ratings[int(mid)] = float(value)
        if len(ratings) < 3:
            st.session_state.pop("recs", None)
            st.warning(f"Please rate at least 3 movies (you rated {len(ratings)}).")
        else:
            strategy = STRATEGIES[strategy_label]
            st.session_state["recs"] = {
                "df": rec.recommend_from_ratings(ratings, strategy=strategy, n=10),
                "cluster": int(rec.cluster_for_ratings(ratings)),
                "strategy": strategy,
                "n_rated": len(ratings),
            }

    result = st.session_state.get("recs")
    if result:
        names = list(rec.cl["names"])
        cid = result["cluster"]
        if 0 <= cid < len(names):
            st.success(f"You look like: **{names[cid]}**  (based on your {result['n_rated']} ratings)")
        st.caption(rec.explain(result["strategy"]))
        if result["strategy"] == "hybrid":
            st.caption("Relative score = average of the CF and content z-scores: it ranks movies against each "
                       "other and is not a star rating.")
        movie_cards(result["df"], SCORE_LABELS.get(result["strategy"]))


def _profile_picker(rec) -> None:
    cl = rec.cl
    names = list(cl["names"])
    top = cl.get("top_genres") or [[] for _ in names]
    labels = [f"{n} ({_genres_pp(t, 2)})" if len(t) else n for n, t in zip(names, top)]
    cid = st.selectbox("Taste profile", range(len(names)), format_func=lambda c: labels[c], key="profile_pick")
    st.caption("Cold-start for a brand-new user: no ratings needed, we serve the profile's favourites.")
    movie_cards(rec.recommend_for_cluster(int(cid), 10), SCORE_LABELS["cluster"])


def tab_recommend(rec) -> None:
    st.subheader("Get recommendations")
    mode = st.radio("How should we learn your taste?", ["Rate movies", "Pick a taste profile"],
                    horizontal=True, key="rec_mode")
    if mode == "Rate movies":
        _rating_form(rec)
    else:
        _profile_picker(rec)


# ----------------------------------------------------------------------------- tab 2
def tab_profiles(rec) -> None:
    st.subheader("Explore taste profiles")
    cl = rec.cl                       # the Recommender's own clusters (same ids as cluster_for_ratings)
    names = list(cl["names"])
    st.caption(
        "K-means on the matrix-factorization user vectors (cosine geometry). Each dot is a user, "
        "projected to 2-D with PCA; colours are taste profiles named after their most over-represented genres."
    )

    labels, xy = np.asarray(cl["labels"]), np.asarray(cl["pca_2d"])
    rng = np.random.default_rng(0)
    idx = rng.choice(len(labels), size=min(3000, len(labels)), replace=False)
    scatter = pd.DataFrame({"PC1": xy[idx, 0], "PC2": xy[idx, 1],
                            "profile": [names[c] if 0 <= c < len(names) else "?" for c in labels[idx]]})
    fig = px.scatter(scatter, x="PC1", y="PC2", color="profile", opacity=0.6,
                     category_orders={"profile": names}, height=480)
    fig.update_traces(marker={"size": 5})
    fig.update_layout(legend_title_text="", margin={"l": 0, "r": 0, "t": 10, "b": 0})
    st.plotly_chart(fig)

    sizes = [int(s) for s in cl.get("sizes", [])]
    total = max(sum(sizes), 1)
    top = cl.get("top_genres") or [[] for _ in names]
    table = pd.DataFrame({
        "Profile": names,
        "Users": sizes,
        "Share": [f"{100 * s / total:.0f}%" for s in sizes],
        "Over-represented genres (vs all users)": [_genres_pp(t, 3) for t in top],
    })
    st.dataframe(table, hide_index=True)

    cid = st.selectbox("Top movies for profile", range(len(names)), format_func=lambda c: names[c],
                       key="explore_pick")
    movie_cards(rec.recommend_for_cluster(int(cid), 10), SCORE_LABELS["cluster"])

    figs = [p for p in (FIGURES_DIR / "elbow.png", FIGURES_DIR / "silhouette.png") if p.exists()]
    if figs:
        st.caption("How k was chosen: silhouette decides (elbow shown for transparency).")
        for col, path in zip(st.columns(len(figs)), figs):
            col.image(str(path), width="stretch")


# ----------------------------------------------------------------------------- tab 3
def simulate(env_dict: dict, policies: list[str], rounds: int, epsilon: float, alpha: float, seed: int) -> dict:
    """Run each selected policy (bandit.POLICY_NAMES keys) on the same simulator and seed, exactly as
    bandit.run_all does for one seed; returns per-policy curves keyed by display label."""
    n_arms, dim = np.asarray(env_dict["expected"]).shape[1], np.asarray(env_dict["contexts"]).shape[1]
    cfg = dataclasses.replace(get_config(), bandit_epsilon=epsilon, bandit_alpha=alpha)
    pols = bandit.make_policies(cfg, n_arms, dim, seed)
    out = {}
    for key in policies:
        env = bandit.BanditEnv(env_dict, np.random.default_rng(seed))
        res = bandit.run_simulation(env, pols[key], rounds)
        out[POLICY_LABELS.get(key, key)] = {
            "cum_reward": np.cumsum(res["reward"]),
            "cum_regret": np.cumsum(res["regret"]),
            "arm_counts": np.bincount(np.asarray(res["arms"], dtype=int), minlength=n_arms),
        }
    return out


def _line_chart(results: dict, key: str, title: str) -> go.Figure:
    fig = go.Figure()
    for name, res in results.items():
        y = res[key]
        fig.add_trace(go.Scatter(x=np.arange(1, len(y) + 1), y=y, mode="lines", name=name))
    fig.update_layout(title=title, xaxis_title="round", height=360, margin={"l": 0, "r": 0, "t": 40, "b": 0},
                      legend={"orientation": "h", "y": -0.25})
    return fig


def _defaults_note(defaults: dict) -> str:
    """One line saying where the default epsilon / alpha come from."""
    values = f"epsilon = {defaults['epsilon']:g}, alpha = {defaults['alpha']:g}"
    if defaults["source"] == "tuned":
        return (f"Defaults are the tuned values ({values}): each was picked by lowest cumulative regret on "
                f"{defaults['tuned_on_users']:,} simulated users disjoint from the users evaluated here.")
    if defaults["source"] == "results":
        return f"Defaults are the values used in the last training run ({values})."
    return f"Defaults come from the config ({values}); retrain to get tuned values."


def tab_bandit() -> None:
    st.subheader("Bandit Lab")
    st.caption(
        "Each round a simulated user arrives; the policy picks which recommender serves them a top-10 list "
        "(CF, content, taste-profile favourites or trending) and observes the share of items clicked. "
        "LinUCB uses the user's context (activity, rating style, taste profile); epsilon-greedy and random do not."
    )
    env_dict = bandit_env()
    if env_dict is None:
        st.info("The bandit simulator has not been built yet (models/bandit_env.npz missing). "
                "Run `python scripts/train_all.py` to create it.")
        return
    arm_names = [str(a) for a in env_dict.get("arm_names", getattr(bandit, "ARMS", []))]

    cfg = get_config()
    defaults = bandit_defaults()
    default_eps = round(min(max(defaults["epsilon"], 0.0), 0.5), 2)
    alpha_options = sorted({*ALPHA_OPTIONS, *getattr(cfg, "bandit_alpha_grid", ()), defaults["alpha"]})
    with st.form("bandit_form"):
        c1, c2 = st.columns(2)
        default_rounds = int(min(max(round(cfg.bandit_rounds / 100) * 100, 100), 3000))
        rounds = c1.slider("Rounds", 100, 3000, default_rounds, step=100, key="bandit_rounds")
        seed = c1.number_input("Seed", min_value=0, max_value=10_000, value=0, step=1, key="bandit_seed")
        epsilon = c2.slider("Epsilon (epsilon-greedy exploration rate)", 0.0, 0.5, default_eps, step=0.01,
                            key="bandit_epsilon")
        alpha = c2.select_slider("Alpha (LinUCB exploration bonus)", options=alpha_options,
                                 value=defaults["alpha"], format_func=lambda a: f"{a:g}", key="bandit_alpha")
        chosen = st.multiselect("Policies", list(bandit.POLICY_NAMES), default=list(bandit.POLICY_NAMES),
                                format_func=lambda k: POLICY_LABELS.get(k, k), key="bandit_policies")
        run = st.form_submit_button("Run simulation", type="primary")
    st.caption(_defaults_note(defaults))

    if run:
        if not chosen:
            st.warning("Select at least one policy.")
        else:
            with st.spinner("Simulating..."):
                st.session_state["bandit_results"] = simulate(env_dict, chosen, int(rounds), float(epsilon),
                                                              float(alpha), int(seed))

    results = st.session_state.get("bandit_results")
    if not results:
        st.caption("Press **Run simulation** to compare policies.")
        return
    c1, c2 = st.columns(2)
    c1.plotly_chart(_line_chart(results, "cum_reward", "Cumulative reward"))
    c2.plotly_chart(_line_chart(results, "cum_regret", "Cumulative regret (lower is better)"))
    counts = pd.DataFrame([
        {"policy": name, "arm": arm_names[a] if a < len(arm_names) else str(a), "times chosen": int(n)}
        for name, res in results.items() for a, n in enumerate(res["arm_counts"])
    ])
    fig = px.bar(counts, x="arm", y="times chosen", color="policy", barmode="group", height=320)
    fig.update_layout(margin={"l": 0, "r": 0, "t": 10, "b": 0}, legend_title_text="")
    st.plotly_chart(fig)
    probs = " / ".join(f"{cfg.click_probs[r]:.2f}" for r in sorted(cfg.click_probs))
    st.caption(
        f"Clicks are simulated from held-out test ratings: p(click) = {probs} for a 1-5 star rating, "
        f"{cfg.click_unrated:.2f} for an unrated movie. Regret = expected clicks of the best arm for that user "
        "minus expected clicks of the chosen arm."
    )


# ----------------------------------------------------------------------------- tab 4
PIPELINE_MD = """
1. **Anomaly detection first.** A Gaussian density model on per-user behaviour features (volume, rating
   style, speed) flags bot-like accounts; epsilon is tuned by F1 on synthetic bots. Flagged users are
   excluded from training so they cannot distort the models.
2. **Collaborative filtering (core).** Matrix factorization (user/item vectors + biases, regularized MSE,
   PyTorch + Adam, early stopping on validation RMSE), compared against a bias baseline and popularity.
3. **Two-tower content model (cold-start movies).** Genres, year and title embeddings for movies; rating-derived
   profiles for users. It can score movies nobody has rated yet, which MF cannot.
4. **Taste profiles (cold-start users).** K-means on the MF user vectors gives named segments, so a new
   user can pick a profile instead of rating anything.
5. **Contextual bandit.** LinUCB learns which recommender to use for which kind of user, scored by a click
   simulator built from held-out ratings, against epsilon-greedy and random.
"""
BANDIT_TUNED_MD = """   Exploration settings are tuned on simulated users disjoint from the evaluated ones.
"""
DQN_STEP_MD = """6. **DQN (stretch goal).** A small deep Q-network learns in which order to show 5 movies in a session,
   where later slots and repeated genres lower the click probability.
"""
PIPELINE_FOOTER_MD = """
The app runs on **numpy only**: weights are exported to `.npz` and new users are folded in with closed-form
ridge regression against the frozen movie vectors.
"""


def _ranking_table(rm: dict) -> pd.DataFrame | None:
    rows = []
    for key, label in MODEL_LABELS.items():
        m = rm.get(key)
        if m:
            rows.append({"Model": label, "RMSE": m.get("rmse"), "P@10": m.get("precision_at_10"),
                         "R@10": m.get("recall_at_10"), "NDCG@10": m.get("ndcg_at_10")})
    return pd.DataFrame(rows) if rows else None


def _cold_table(rm: dict) -> pd.DataFrame | None:
    cold = (rm.get("two_tower") or {}).get("cold")
    base = rm.get("cold_baselines") or {}
    rows = []
    if cold:
        rows.append({"Model": "Two-tower (metadata only)", "RMSE": cold.get("rmse"),
                     "P@10": cold.get("precision_at_10"), "R@10": cold.get("recall_at_10"),
                     "NDCG@10": cold.get("ndcg_at_10")})
    if base.get("user_bias"):
        rows.append({"Model": "User-bias baseline (best CF can do)", "RMSE": base["user_bias"].get("rmse"),
                     "P@10": None, "R@10": None, "NDCG@10": None})
    if base.get("random"):
        r = base["random"]
        rows.append({"Model": "Random ranking", "RMSE": None, "P@10": r.get("precision_at_10"),
                     "R@10": r.get("recall_at_10"), "NDCG@10": r.get("ndcg_at_10")})
    return pd.DataFrame(rows) if rows else None


def _show_table(df: pd.DataFrame) -> None:
    cols = [c for c in df.columns if c != "Model"]
    st.dataframe(df.style.format({c: lambda v: _fmt(v) for c in cols}, na_rep="n/a"), hide_index=True)


def _data_sentence(metrics: dict) -> str:
    d = metrics.get("data") or {}
    if not d:
        return f"{metrics.get('dataset', 'MovieLens-1M')}; split: {metrics.get('split', 'n/a')}."
    return (
        f"{metrics.get('dataset', 'MovieLens-1M')}: {d.get('n_users', 0):,} users, {d.get('n_items', 0):,} movies "
        f"({d.get('n_cold_items', 0):,} held out as cold-start). Split: {metrics.get('split', 'n/a')} "
        f"({d.get('n_train', 0):,} train / {d.get('n_val', 0):,} val / {d.get('n_test', 0):,} test ratings). "
        f"{d.get('n_flagged_users', 0):,} users flagged by anomaly detection were excluded from training. "
        f"Ranking metrics use a fixed seeded sample of {d.get('n_eval_users', 0):,} test users; "
        f"a held-out rating >= {RELEVANT_THRESHOLD} counts as relevant."
    )


def _headline(metrics: dict) -> None:
    rm = metrics.get("rating_models") or {}
    tiles = []
    mf, base = rm.get("cf_mf") or {}, rm.get("bias_baseline") or {}
    if mf.get("rmse") is not None:
        delta = None if base.get("rmse") is None else f"{mf['rmse'] - base['rmse']:+.4f} vs bias baseline"
        tiles.append(("MF test RMSE", _fmt(mf["rmse"]), delta))
    cold = (rm.get("two_tower") or {}).get("cold") or {}
    if cold.get("ndcg_at_10") is not None:
        tiles.append(("Cold-start NDCG@10", _fmt(cold["ndcg_at_10"]), None))
    an = metrics.get("anomaly") or {}
    if an.get("test_f1") is not None:
        tiles.append(("Bot detection F1 (test)", _fmt(an["test_f1"]), None))
    pol = ((metrics.get("bandit") or {}).get("policies") or {})
    if (pol.get("linucb") or {}).get("cum_regret") is not None and (pol.get("random") or {}).get("cum_regret"):
        ratio = pol["linucb"]["cum_regret"] / pol["random"]["cum_regret"]
        tiles.append(("LinUCB regret vs random", f"{100 * ratio:.0f}%", None))
    if tiles:
        for col, (label, value, delta) in zip(st.columns(len(tiles)), tiles):
            col.metric(label, value, delta, delta_color="inverse")


def _bandit_note(bd: dict) -> str:
    """Tuned hyperparameters and the share of simulated users without any click signal."""
    parts = []
    params = bd.get("params") or {}
    if params.get("epsilon") is not None and params.get("alpha") is not None:
        txt = f"Epsilon {params['epsilon']:g}, alpha {params['alpha']:g}"
        if params.get("tuned_on_users"):
            txt += f" (tuned on {int(params['tuned_on_users']):,} disjoint simulated users)"
        parts.append(txt + ".")
    if bd.get("frac_users_no_signal") is not None:
        parts.append(f"{100 * float(bd['frac_users_no_signal']):.0f}% of simulated users have no held-out "
                     "test rating among any arm's top-10, so every arm looks the same for them.")
    return " ".join(parts)


def _dqn_section(dqn: dict) -> None:
    """Small table of expected clicks per 5-slot session (metrics["dqn"], spec 18.5)."""
    st.markdown(f"**DQN session ordering** (stretch goal; expected clicks per {dqn.get('session_len', 5)}-slot "
                "session on held-out users)")
    rows = [{"Model": label, "Expected clicks": dqn.get(key)} for key, label in DQN_LABELS.items()
            if dqn.get(key) is not None]
    if rows:
        _show_table(pd.DataFrame(rows))
    sizes = [f"{dqn[k]:,} {label}" for k, label in (("n_train_users", "train users"),
                                                    ("n_eval_users", "eval users"), ("episodes", "episodes"))
             if dqn.get(k) is not None]
    if sizes:
        st.caption("DQN: " + ", ".join(sizes) + ". Reported as measured, even where the heuristic wins.")


def _results(metrics: dict) -> None:
    if metrics.get("fast"):
        st.warning("FAST MODE: these numbers come from a `--fast` smoke run and are not final.")
    _headline(metrics)
    st.caption(_data_sentence(metrics))
    rm = metrics.get("rating_models") or {}

    table = _ranking_table(rm)
    if table is not None:
        st.markdown("**Warm movies** (test split)")
        _show_table(table)
        st.caption("Popularity has no rating prediction (RMSE n/a). It can beat MF on top-N in MovieLens; "
                   "MF here is tuned for RMSE. Reported as measured.")
    cold = _cold_table(rm)
    if cold is not None:
        st.markdown("**Cold-start movies** (never seen in training)")
        _show_table(cold)

    an = metrics.get("anomaly")
    if an:
        st.markdown("**Anomaly detection** (synthetic bots + held-out real users)")
        rows = [{"Model": "Deployed detector", "Precision": an.get("test_precision"),
                 "Recall": an.get("test_recall"), "F1": an.get("test_f1")}]
        for key, label in ABLATION_LABELS.items():
            abl = an.get(key)
            if abl:
                rows.append({"Model": label, "Precision": abl.get("test_precision"),
                             "Recall": abl.get("test_recall"), "F1": abl.get("test_f1")})
        _show_table(pd.DataFrame(rows))
        per_type = an.get("per_bot_type_recall") or {}
        if per_type:
            st.caption("Recall per bot type: " + ", ".join(f"{k} {_fmt(v)}" for k, v in per_type.items())
                       + f". Real users flagged and excluded: {an.get('n_real_flagged_all', 'n/a')}.")

    cl = metrics.get("clustering")
    if cl:
        sil = cl.get("silhouette")
        note = (" (low: the profiles are a coarse segmentation of a continuous taste space)"
                if sil is not None and float(sil) < 0.25 else "")
        st.markdown(f"**Taste profiles:** k = {cl.get('k')}, silhouette {_fmt(sil)}{note}.")

    bd = metrics.get("bandit")
    if bd:
        st.markdown(f"**Contextual bandit** ({bd.get('rounds', 0):,} rounds x {bd.get('n_seeds', 0)} seeds)")
        rows = [{"Model": POLICY_LABELS.get(name, name), "Mean reward": p.get("mean_reward"), "Cumulative regret": p.get("cum_regret"),
                 "Regret std": p.get("cum_regret_std")} for name, p in (bd.get("policies") or {}).items()]
        if rows:
            _show_table(pd.DataFrame(rows))
        note = _bandit_note(bd)
        if note:
            st.caption(note)

    dqn = metrics.get("dqn")
    if dqn:
        _dqn_section(dqn)
    else:
        st.caption("DQN stretch goal: not shipped.")
    st.caption(f"Generated by scripts/evaluate.py at {metrics.get('generated_at', 'n/a')}.")


ABLATION_LABELS = {
    "ablation_plus_item_dev": "+ item-deviation feature (ablation)",
    "ablation_5_features": "5 features (ablation)",       # older metrics.json files
}
DQN_LABELS = {"dqn": "DQN (greedy policy)", "greedy_cf": "CF order", "diversity_heuristic": "Diversity heuristic",
              "random": "Random order"}

FEATURE_HELP = {
    "log_count": "log(1 + number of ratings)",
    "mean_rating": "average star rating",
    "var_rating": "variance of the star ratings",
    "pct_extreme": "share of 1- and 5-star ratings",
    "log_ratings_per_day": "log(1 + ratings / days between first and last rating, min 1 day)",
    "mean_abs_item_dev": "mean |rating - movie's average rating|",
}
# A push bot: 150 ratings within one hour, all 1 or 5 stars (mean 4.9 -> variance 3.9 * 0.1 = 0.39),
# far from the crowd's opinion.
PUSH_BOT = {"log_count": float(np.log1p(150)), "mean_rating": 4.9, "var_rating": 0.39, "pct_extreme": 1.0,
            "log_ratings_per_day": float(np.log1p(150)), "mean_abs_item_dev": 1.4}


def _anomaly_demo() -> None:
    st.markdown("#### Try the bot detector")
    model = anomaly_model()
    if model is None:
        st.info("Anomaly model not found (models/anomaly.npz). Run `python scripts/train_all.py`.")
        return
    # One input per feature the saved model was trained on, in its stored order.
    features = [str(f) for f in model.get("features", [])]
    if len(features) != len(np.atleast_1d(model["feat_mean"])):
        st.info("The saved anomaly model has inconsistent feature names; retrain to use the demo.")
        return
    st.caption("A user is flagged when the log-density of their behaviour features falls below epsilon. "
               "Start from an average real user or a push bot and tweak the inputs.")
    typical = {f: float(v) for f, v in zip(features, np.atleast_1d(model["feat_mean"]))}
    preset = st.radio("Start from", ["Average real user", "Push bot"], horizontal=True, key="anom_preset")
    base = typical if preset == "Average real user" else {f: PUSH_BOT.get(f, typical[f]) for f in features}
    cols = st.columns(3)
    values = [
        cols[j % 3].number_input(f, value=round(base[f], 3), step=0.05, format="%.3f",
                                 help=FEATURE_HELP.get(f), key=f"anom_{preset}_{f}")
        for j, f in enumerate(features)
    ]
    score, eps = anomaly.score_user(model, np.array(values, dtype=np.float64)), float(model["eps"])
    c1, c2 = st.columns(2)
    c1.metric("Log-density score", f"{score:.2f}")
    c2.metric("Threshold epsilon", f"{eps:.2f}")
    if score < eps:
        st.error("Flagged as anomalous: this user would be excluded from training.")
    else:
        st.success("Looks like a normal user.")


def tab_how() -> None:
    st.subheader("How it works")
    metrics = load_metrics()
    has_dqn = bool(metrics and metrics.get("dqn"))
    tuned = bool((((metrics or {}).get("bandit") or {}).get("params") or {}).get("tuned_on_users"))
    st.markdown(PIPELINE_MD + (BANDIT_TUNED_MD if tuned else "") + (DQN_STEP_MD if has_dqn else "")
                + PIPELINE_FOOTER_MD)
    st.markdown("#### Results")
    if metrics is None:
        st.info("reports/metrics.json not found. Run `python scripts/evaluate.py` to produce the results.")
    else:
        _results(metrics)
    figs = [(p, cap) for p, cap in ((FIGURES_DIR / "cf_training.png", "MF training (RMSE per epoch)"),
                                    (FIGURES_DIR / "anomaly_scores.png", "Anomaly scores: real users vs bots"),
                                    (FIGURES_DIR / "bandit_regret.png", "Bandit cumulative regret"),
                                    (FIGURES_DIR / "dqn_training.png", "DQN training reward"))
            if p.exists() and (has_dqn or p.name != "dqn_training.png")]
    if figs:
        st.markdown("#### Figures")
        for col, (path, cap) in zip(st.columns(len(figs)), figs):
            col.image(str(path), caption=cap, width="stretch")
    _anomaly_demo()


# ----------------------------------------------------------------------------- page
def sidebar() -> None:
    with st.sidebar:
        st.title("CineMatch")
        st.markdown(
            "A MovieLens-1M recommender that combines **anomaly detection**, **matrix factorization**, a "
            "**two-tower content model**, **K-means taste profiles** and a **contextual bandit**."
        )
        st.markdown(f"[Source code on GitHub]({REPO_URL})")
        if os.environ.get("TMDB_API_KEY"):
            st.caption("Posters: TMDB (API key found).")
        else:
            st.caption("Posters: placeholders (set TMDB_API_KEY for real posters).")
        st.caption("Movie data: MovieLens-1M (GroupLens), used under its terms; not redistributed.")


def main() -> None:
    st.set_page_config(page_title="CineMatch", page_icon=":material/movie:", layout="wide")
    sidebar()
    st.title("CineMatch")
    st.caption("Movie recommendations from 1M ratings: collaborative filtering, content, taste profiles and bandits.")
    rec = load_recommender()
    tabs = st.tabs(["Get Recommendations", "Explore Taste Profiles", "Bandit Lab", "How It Works"])
    with tabs[0]:
        if rec is None:
            st.info("Recommendations need trained models (see the instructions above).")
        else:
            tab_recommend(rec)
    with tabs[1]:
        if rec is None:
            st.info("Taste profiles need trained models (see the instructions above).")
        else:
            tab_profiles(rec)
    with tabs[2]:
        tab_bandit()
    with tabs[3]:
        tab_how()


main()
