"""Streamlit UI for the agent-orchestrated music recommender.
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import copy
import html
import json
import secrets
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from src.recommender import config
from src.recommender.core.io import load_songs
from src.recommender.core.rerank import SIGNAL_LABELS
from src.recommender.llm import agent
from src.recommender.main import PROFILES
from src.recommender.rag import get_retriever
from src.recommender import spotify
from src.recommender.spotify import auth as spotify_auth
from src.recommender.spotify.taste import summary_lines


st.set_page_config(
    page_title="Agentic Music Recommender",
    page_icon="🎵",
    layout="wide",
)


FLAG_LABELS = {
    "low_result_count": "Low result count",
    "constraint_violated": "Constraint violated",
    "intent_mismatch": "Intent mismatch",
    "looks_good": "Looks good",
}

FLAG_EXPLANATIONS = {
    "low_result_count": "The retrieval step returned fewer matches than expected for the query.",
    "constraint_violated": "At least one candidate appears to break a hard constraint from the plan.",
    "intent_mismatch": "The returned songs do not really fit the user's request as a set.",
    "looks_good": "The system believes the candidates are a strong match for the query.",
}


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg-deep: #170f2b;
            --bg-mid: #26164a;
            --panel: rgba(33, 20, 62, 0.82);
            --panel-soft: rgba(255, 255, 255, 0.09);
            --text-main: #f7f2ff;
            --text-soft: #d8c9f6;
            --text-dark: #25153f;
            --accent-violet: #a970ff;
            --accent-pink: #ff79c6;
            --accent-lilac: #d8b4fe;
            --accent-ice: #fdf7ff;
            --border-soft: rgba(255, 255, 255, 0.14);
        }
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(255, 121, 198, 0.35), transparent 28%),
                radial-gradient(circle at top right, rgba(169, 112, 255, 0.28), transparent 24%),
                radial-gradient(circle at bottom right, rgba(216, 180, 254, 0.18), transparent 20%),
                linear-gradient(180deg, #120b24 0%, #1b1034 44%, #241144 100%);
            color: var(--text-main);
        }
        .stApp, .stApp p, .stApp label, .stApp li {
            color: var(--text-main);
        }
        .block-container {
            padding-top: 2rem;
        }
        [data-testid="stSidebar"] {
            background:
                linear-gradient(180deg, rgba(20, 11, 39, 0.96), rgba(33, 18, 63, 0.96));
            border-right: 1px solid var(--border-soft);
        }
        [data-testid="stSidebar"] * {
            color: var(--text-main);
        }
        [data-testid="stMetric"] {
            background: var(--panel-soft);
            border: 1px solid var(--border-soft);
            border-radius: 18px;
            padding: 0.85rem 1rem;
            box-shadow: 0 12px 28px rgba(7, 3, 20, 0.22);
        }
        [data-testid="stMetricLabel"] p,
        [data-testid="stMetricValue"] {
            color: var(--text-main);
        }
        .stAlert {
            background: rgba(255, 255, 255, 0.08);
            color: var(--text-main);
            border: 1px solid var(--border-soft);
        }
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4,
        .stMarkdown p, .stMarkdown li, .stCaption {
            color: var(--text-main);
        }
        .stSubheader, .stHeader {
            color: var(--accent-ice);
        }
        [data-testid="stExpander"] {
            background: linear-gradient(180deg, rgba(255, 249, 255, 0.96), rgba(245, 234, 255, 0.94));
            border: 1px solid rgba(216, 180, 254, 0.42);
            border-radius: 16px;
        }
        [data-testid="stExpander"] summary, [data-testid="stExpander"] summary * {
            color: var(--text-dark) !important;
            font-weight: 700;
        }
        [data-testid="stExpanderDetails"] {
            color: var(--text-dark);
        }
        [data-testid="stExpanderDetails"] p,
        [data-testid="stExpanderDetails"] div,
        [data-testid="stExpanderDetails"] span {
            color: var(--text-dark) !important;
        }
        [data-testid="stDataFrame"] {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 18px;
            border: 1px solid var(--border-soft);
            padding: 0.2rem;
        }
        .stTextInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"],
        .stNumberInput input {
            background: rgba(255, 255, 255, 0.08);
            color: var(--text-main);
            border-radius: 14px;
        }
        .stButton > button {
            background: linear-gradient(135deg, var(--accent-violet), var(--accent-pink));
            color: white;
            border: none;
            border-radius: 14px;
            font-weight: 700;
        }
        .stButton > button:hover {
            background: linear-gradient(135deg, #bb8bff, #ff92d0);
            color: white;
        }
        .hero {
            padding: 1.4rem 1.6rem;
            border-radius: 24px;
            background:
                linear-gradient(135deg, rgba(116, 59, 189, 0.92), rgba(255, 121, 198, 0.88));
            color: white;
            box-shadow: 0 18px 45px rgba(12, 5, 24, 0.35);
            margin-bottom: 1rem;
            border: 1px solid rgba(255, 255, 255, 0.16);
        }
        .hero h1 {
            margin: 0 0 0.35rem 0;
            font-size: 2.2rem;
            color: white;
        }
        .hero p {
            margin: 0;
            max-width: 56rem;
            line-height: 1.5;
            color: rgba(255, 255, 255, 0.94);
        }
        .rec-card {
            background: linear-gradient(180deg, rgba(255, 247, 255, 0.96), rgba(246, 236, 255, 0.94));
            border: 1px solid rgba(216, 180, 254, 0.45);
            border-radius: 20px;
            padding: 1rem 1.05rem;
            box-shadow: 0 10px 30px rgba(8, 3, 20, 0.22);
            height: 100%;
        }
        .rec-card, .rec-card div, .rec-card span, .rec-card p {
            color: var(--text-dark) !important;
        }
        .rec-kicker {
            color: #a43ee3;
            font-size: 0.75rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.35rem;
        }
        .rec-title {
            font-size: 1.15rem;
            font-weight: 700;
            color: var(--text-dark);
            margin-bottom: 0.25rem;
        }
        .rec-meta {
            color: #5f427f;
            margin-bottom: 0.7rem;
        }
        .pill-row {
            margin: 0.6rem 0 0.8rem 0;
        }
        .pill {
            display: inline-block;
            margin: 0 0.35rem 0.35rem 0;
            padding: 0.22rem 0.6rem;
            border-radius: 999px;
            background: linear-gradient(135deg, #f2dcff, #ffd7ee);
            color: #5d1d74;
            font-size: 0.78rem;
            font-weight: 600;
        }
        .trace-box {
            background: rgba(255, 255, 255, 0.07);
            border: 1px solid var(--border-soft);
            border-radius: 18px;
            padding: 0.9rem 1rem;
            margin-bottom: 0.85rem;
            box-shadow: 0 12px 28px rgba(8, 3, 20, 0.18);
        }
        .trace-kind {
            font-size: 0.78rem;
            font-weight: 700;
            color: #ff9bd7;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.45rem;
        }
        .rec-body {
            background: rgba(255, 255, 255, 0.74);
            border: 1px solid rgba(216, 180, 254, 0.35);
            border-radius: 14px;
            padding: 0.8rem 0.9rem;
            color: #2b1846 !important;
            margin: 0.65rem 0 0.8rem 0;
            font-weight: 500;
            line-height: 1.6;
        }
        .flag-legend {
            background: rgba(255, 255, 255, 0.07);
            border: 1px solid var(--border-soft);
            border-radius: 18px;
            padding: 1rem 1rem 0.55rem 1rem;
            margin-top: 0.9rem;
            box-shadow: 0 12px 28px rgba(8, 3, 20, 0.18);
        }
        .flag-item {
            margin-bottom: 0.8rem;
        }
        .flag-name {
            color: #ffd2f0;
            font-weight: 700;
            margin-bottom: 0.15rem;
        }
        .flag-desc {
            color: var(--text-soft);
            line-height: 1.45;
        }
        .status-box {
            background: rgba(255, 255, 255, 0.07);
            border: 1px solid var(--border-soft);
            border-radius: 18px;
            padding: 1rem 1rem 0.9rem 1rem;
            margin-top: 0.9rem;
            box-shadow: 0 12px 28px rgba(8, 3, 20, 0.18);
        }
        .status-title {
            color: #ffd2f0;
            font-weight: 700;
            margin-bottom: 0.3rem;
        }
        .status-text {
            color: var(--text-soft);
            line-height: 1.5;
        }
        .cite {
            display: inline-block;
            margin-left: 0.15rem;
            padding: 0 0.32rem;
            border-radius: 0.4rem;
            font-size: 0.72rem;
            font-weight: 700;
            vertical-align: super;
            color: var(--text-dark);
            background: var(--accent-lilac);
        }
        .evidence-item {
            margin: 0.35rem 0;
            font-size: 0.9rem;
            color: var(--text-soft);
        }
        .evidence-item b { color: var(--text-main); }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def _load_catalog(csv_path: str) -> List[Dict[str, Any]]:
    return load_songs(csv_path)


# Results are cached on primitive inputs, so re-running the same query (or any widget
# interaction that triggers a Streamlit rerun) costs nothing.
@st.cache_resource(show_spinner="Loading retrieval index...")
def _warm_retriever(csv_path: str) -> bool:
    """Load the embedding model and catalog vectors once per server, not per query."""
    get_retriever(_load_catalog(csv_path)).search("warm up")
    return True


# Profiles are passed as JSON so the cache key covers a Spotify profile's contents.
@st.cache_data(show_spinner=False, ttl=3600, max_entries=128)
def _run_ranked(query: str, profile_json: str, k: int, songs_path: str,
                skip_recent: bool) -> agent.AgentResult:
    songs = _load_catalog(songs_path)
    profile = json.loads(profile_json)
    if skip_recent and profile.get("recently_played"):
        recent = set(profile["recently_played"])
        songs = [s for s in songs if s["id"] not in recent]
    return agent.run(query, profile, songs, k=k, use_llm_explainer=False)


@st.cache_data(show_spinner=False, ttl=3600, max_entries=128)
def _run_explained(query: str, profile_json: str, k: int, songs_path: str,
                   skip_recent: bool) -> agent.AgentResult:
    result = copy.deepcopy(_run_ranked(query, profile_json, k, songs_path, skip_recent))
    return agent.explain_result(result)


# ── Spotify ──────────────────────────────────────────────────────────────────

@st.cache_resource
def _pending_logins() -> Dict[str, str]:
    """state -> PKCE verifier. Server-wide, because the OAuth redirect opens a new
    browser session that can't see the old session_state."""
    return {}


@st.cache_data(show_spinner="Reading your Spotify history...", ttl=1800)
def _spotify_profile(access_token: str, songs_path: str) -> Dict[str, Any]:
    store = spotify.TokenStore()
    tokens = store.load()
    client = spotify.SpotifyClient(
        tokens if tokens and tokens.access_token == access_token
        else spotify.Tokens(access_token, "", 0),
        on_refresh=lambda t: _refresh_and_save(store, t),
    )
    history = client.fetch_history()
    return spotify.build_profile(history, _load_catalog(songs_path))


def _refresh_and_save(store, tokens):
    fresh = spotify_auth.refresh(tokens)
    store.save(fresh)
    return fresh


def _handle_spotify_callback() -> None:
    params = st.query_params
    if "code" not in params and "error" not in params:
        return
    if "error" in params:
        st.error(f"Spotify login was cancelled: {params['error']}")
        st.query_params.clear()
        return
    verifier = _pending_logins().pop(params.get("state", ""), None)
    if verifier is None:
        st.error("Spotify login expired. Click Connect Spotify again.")
        st.query_params.clear()
        return
    try:
        tokens = spotify_auth.exchange_code(params["code"], verifier)
    except spotify.SpotifyAuthError as exc:
        st.error(str(exc))
    else:
        spotify.TokenStore().save(tokens)
    st.query_params.clear()
    st.rerun()


def _render_spotify(songs_path: str) -> Dict[str, Any] | None:
    st.sidebar.subheader("Spotify")
    if not spotify.is_configured():
        st.sidebar.caption("Set `SPOTIFY_CLIENT_ID` in `.env` to personalize with your listening history.")
        return None

    store = spotify.TokenStore()
    tokens = store.valid()
    if tokens is None:
        state = secrets.token_urlsafe(16)
        verifier = spotify_auth.new_verifier()
        _pending_logins()[state] = verifier
        url = spotify_auth.authorize_url(state, verifier)
        st.sidebar.markdown(
            f'<a href="{html.escape(url)}" target="_self">'
            f'<button style="width:100%;padding:0.5rem;border-radius:0.5rem;cursor:pointer;">'
            f"Connect Spotify</button></a>",
            unsafe_allow_html=True,
        )
        return None

    try:
        profile = _spotify_profile(tokens.access_token, songs_path)
    except (spotify.SpotifyAPIError, ValueError) as exc:
        st.sidebar.error(str(exc))
        return None
    for line in summary_lines(profile):
        st.sidebar.caption(line)
    with st.sidebar.expander("Your taste profile"):
        summary = profile["history_summary"]
        st.write(f"Derived: genre **{profile['genre']}**, mood **{profile['mood']}**, "
                 f"energy {profile['energy']}, valence {profile['target_valence']}")
        st.write("Top tracks:")
        for t in summary["top_tracks"][:5]:
            st.caption(t)
        if summary["genre_mix"]:
            st.bar_chart(pd.Series(summary["genre_mix"], name="share"), horizontal=True, height=180)
    if st.sidebar.button("Disconnect Spotify"):
        store.clear()
        _spotify_profile.clear()
        st.rerun()
    return profile


def _render_header() -> None:
    st.markdown(
        """
        <div class="hero">
          <h1>Agentic Music Recommender</h1>
          <p>
            Explore the plan-act-check flow behind each recommendation, inspect
            the retrieved candidates, and see how the system handles queries like
            <strong>“chill study music, no vocals.”</strong>
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _profile_label(profile_key: str, profiles: Dict[str, Dict[str, Any]]) -> str:
    profile = profiles[profile_key]
    if profile_key == "spotify":
        return profile["name"]
    return f"{profile['name']} ({profile['genre']}, {profile['mood']})"


def _candidate_table(result: agent.AgentResult) -> pd.DataFrame:
    rows = []
    for rec in result.recommendations:
        song = rec.song
        rows.append(
            {
                "Title": song["title"],
                "Artist": song["artist"],
                "Genre": song["genre"],
                "Mood": song["mood"],
                "BPM": round(song["tempo_bpm"], 1),
                "Energy": round(song["energy"], 2),
                "Instrumentalness": round(song.get("instrumentalness", 0.0), 2),
                "Popularity": song.get("popularity", 0),
            }
        )
    return pd.DataFrame(rows)


def _render_metrics(result: agent.AgentResult) -> None:
    flags = [flag.value for flag in result.verdict.flags]
    metric_cols = st.columns(5)
    metric_cols[0].metric("Candidates", result.candidate_count)
    metric_cols[1].metric("Recommendations", len(result.recommendations))
    metric_cols[2].metric("Confidence", f"{result.verdict.confidence:.2f}")
    metric_cols[3].metric("Flags", len(flags))
    metric_cols[4].metric("Latency", f"{result.timings_ms.get('total', 0) / 1000:.2f}s")


def _render_timings(result: agent.AgentResult) -> None:
    timings = {k: v for k, v in result.timings_ms.items() if k != "total"}
    if not timings:
        return
    st.subheader("Latency by Stage")
    st.bar_chart(pd.Series(timings, name="ms"), horizontal=True)


def _render_flag_legend(result: agent.AgentResult) -> None:
    flags = [flag.value for flag in result.verdict.flags]
    with st.container(border=True):
        st.markdown("**Flag Guide**")
        if not flags:
            st.write("No active flags. Nothing concerning was flagged for this run.")
            return

        for flag in flags:
            label = FLAG_LABELS.get(flag, flag.replace("_", " ").title())
            description = FLAG_EXPLANATIONS.get(flag, "No description available.")
            st.markdown(f"**{label}**")
            st.write(description)


def _status_summary(result: agent.AgentResult) -> tuple[str, str]:
    flags = [flag.value for flag in result.verdict.flags]
    confidence = result.verdict.confidence

    if not flags:
        return (
            "Overall status",
            "The system did not raise any concerns for this run.",
        )
    if flags == ["looks_good"]:
        return (
            "Overall status",
            "The retrieved songs look like a strong match for the query.",
        )

    parts = []
    for flag in flags:
        description = FLAG_EXPLANATIONS.get(flag)
        if description and description not in parts:
            parts.append(description)

    if result.candidate_count == 0:
        parts.insert(0, "No songs were returned on the final retrieval pass.")

    summary = " ".join(parts) if parts else "The run surfaced some quality concerns."
    summary += f" Confidence: {confidence:.2f}."
    return ("Overall status", summary)


def _render_status_summary(result: agent.AgentResult) -> None:
    title, summary = _status_summary(result)
    st.markdown(
        f"""
        <div class="status-box">
          <div class="status-title">{title}</div>
          <div class="status-text">{summary}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


SOURCE_LABELS = {
    "catalog": "Catalog",
    "wikipedia": "Wikipedia",
    "lastfm": "Last.fm",
    "spotify": "Your Spotify",
}


def _citation_numbers(rec) -> Dict[str, int]:
    """Number evidence in the order claims first cite it: signals and documents alike."""
    numbers: Dict[str, int] = {}
    for claim in rec.claims:
        for eid in claim["evidence_ids"]:
            numbers.setdefault(eid, len(numbers) + 1)
    return numbers


def _claims_html(rec, numbers: Dict[str, int]) -> str:
    parts = []
    for claim in rec.claims:
        chips = "".join(
            f'<span class="cite">{numbers[eid]}</span>' for eid in claim["evidence_ids"] if eid in numbers
        )
        parts.append(f"{html.escape(claim['text'])}{chips}")
    return " ".join(parts) if parts else html.escape(rec.narrative)


def _found_by(rec) -> str:
    r = rec.retrieval
    if not r:
        return ""
    return {"both": "semantic + keyword", "dense": "semantic", "bm25": "keyword"}[r["found_by"]]


def _render_why(rec, numbers: Dict[str, int]) -> None:
    signals = rec.signals
    if signals:
        st.markdown("**Score breakdown**: each bar is weight × value; bars sum to the score.")
        chart = pd.Series(
            {s["label"]: s["contribution"] for s in signals if s["contribution"] > 0}, name="contribution"
        )
        st.bar_chart(chart, horizontal=True, height=min(60 + 28 * len(chart), 320))
        st.dataframe(
            pd.DataFrame([
                {"Signal": s["label"], "Value": s["value"], "Weight": s["weight"],
                 "Contribution": s["contribution"], "Detail": s["detail"]}
                for s in signals
            ]),
            hide_index=True,
            width="stretch",
        )

    r = rec.retrieval
    if r:
        ranks = [f"semantic rank #{r['dense_rank']}"]
        if r.get("bm25_rank"):
            ranks.append(f"keyword rank #{r['bm25_rank']}")
        terms = f" · matched terms: `{'`, `'.join(r['matched_terms'])}`" if r["matched_terms"] else ""
        st.markdown(f"**Retrieval**: found by {_found_by(rec)} ({', '.join(ranks)}){terms}")

    if rec.evidence or numbers:
        st.markdown("**Evidence**")
        signal_by_id = {f"sig:{s['name']}": s for s in signals}
        doc_by_id = {d["doc_id"]: d for d in rec.evidence}
        ordered = list(numbers) + [d["doc_id"] for d in rec.evidence if d["doc_id"] not in numbers]
        for eid in ordered:
            label = f"[{numbers[eid]}]" if eid in numbers else "·"
            if eid in signal_by_id:
                s = signal_by_id[eid]
                body = f"<b>{label} Score signal: {html.escape(s['label'])}</b>: {html.escape(s['detail'])}"
            elif eid in doc_by_id:
                d = doc_by_id[eid]
                src = SOURCE_LABELS.get(d["source"], d["source"])
                link = f' <a href="{html.escape(d["url"])}" target="_blank">source</a>' if d.get("url") else ""
                body = (f"<b>{label} {src}: {html.escape(d['title'])}</b> "
                        f"(similarity {d['score']:.2f}){link}<br>{html.escape(d['text'])}")
            else:
                continue
            st.markdown(f'<div class="evidence-item">{body}</div>', unsafe_allow_html=True)

    if rec.dropped_claims:
        st.markdown("**Dropped claims** (failed citation check)")
        for d in rec.dropped_claims:
            st.caption(f"✗ {d['text']} ({d['reason']})")


def _render_recommendation_cards(result: agent.AgentResult) -> None:
    if not result.recommendations:
        st.warning("No recommendations were returned for this query.")
        return

    cols = st.columns(2)
    for index, rec in enumerate(result.recommendations):
        song = rec.song
        numbers = _citation_numbers(rec)
        found_by = _found_by(rec)
        with cols[index % 2]:
            st.markdown(
                f"""
                <div class="rec-card">
                  <div class="rec-kicker">Recommendation {index + 1}</div>
                  <div class="rec-title">{html.escape(song['title'])}</div>
                  <div class="rec-meta">{html.escape(song['artist'])}</div>
                  <div class="pill-row">
                    <span class="pill">{html.escape(song['genre'])}</span>
                    <span class="pill">{html.escape(song['mood'])}</span>
                    <span class="pill">score {rec.score:.3f}</span>
                    {f'<span class="pill">found by {found_by}</span>' if found_by else ''}
                  </div>
                  <div class="rec-body">{_claims_html(rec, numbers)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            with st.expander("Why this song"):
                _render_why(rec, numbers)


def _render_what_if(result: agent.AgentResult) -> None:
    if not result.recommendations or not result.candidates:
        return
    st.subheader("What If")
    available = [s["name"] for s in result.recommendations[0].signals]
    exclude = st.multiselect(
        "Remove signals and re-rank the same candidates",
        options=available,
        format_func=lambda n: SIGNAL_LABELS.get(n, n),
        key="what_if_exclude",
    )
    if not exclude:
        st.caption(f"{len(result.candidates)} candidates in the pool. Pick a signal to see how the ranking shifts.")
        return
    rows = agent.what_if(result, set(exclude))

    def movement(row) -> str:
        prev = row["previous_rank"]
        if prev is None:
            return "new"
        delta = prev - row["rank"]
        return "same" if delta == 0 else f"up {delta}" if delta > 0 else f"down {-delta}"

    st.dataframe(
        pd.DataFrame([
            {"Rank": r["rank"], "Song": r["title"], "Score": r["score"], "Change": movement(r)}
            for r in rows
        ]),
        hide_index=True,
        width="stretch",
    )


def _render_trace(result: agent.AgentResult) -> None:
    st.subheader("Agent Trace")
    for event in result.trace_events:
        kind = event["kind"]
        with st.container():
            st.markdown('<div class="trace-box">', unsafe_allow_html=True)
            st.markdown(f'<div class="trace-kind">{kind}</div>', unsafe_allow_html=True)
            if kind == "plan":
                st.json(event["plan"])
            elif kind == "check":
                st.json(event["verdict"])
            else:
                st.json({k: v for k, v in event.items() if k not in {"kind", "ts"}})
            st.caption(event["ts"])
            st.markdown("</div>", unsafe_allow_html=True)


def _render_sidebar() -> Dict[str, Any]:
    st.sidebar.header("Run Settings")
    st.session_state.setdefault("songs_path", str(config.SONGS_CSV))
    songs_path = st.session_state["songs_path"]
    spotify_profile = _render_spotify(songs_path)
    profiles = dict(PROFILES)
    if spotify_profile:
        profiles = {"spotify": spotify_profile, **profiles}
    profile_key = st.sidebar.selectbox(
        "Profile",
        options=list(profiles.keys()),
        format_func=lambda key: _profile_label(key, profiles),
        index=0 if spotify_profile else list(profiles.keys()).index("chill_lofi"),
    )
    skip_recent = False
    if profile_key == "spotify":
        skip_recent = st.sidebar.checkbox("Skip songs I played recently", value=False)
    query = st.sidebar.text_area(
        "Query",
        value="chill study music, no vocals",
        height=110,
    )
    k = st.sidebar.slider("Recommendations", min_value=1, max_value=10, value=5)
    use_llm_explainer = st.sidebar.checkbox("Use LLM explainer", value=True)
    songs_path = st.sidebar.text_input("Songs CSV", key="songs_path")
    run_clicked = st.sidebar.button("Generate Recommendations", width="stretch", type="primary")

    st.sidebar.markdown("---")
    profile = profiles[profile_key]
    st.sidebar.caption(
        f"Profile: {profile['name']} | genre={profile['genre']} | mood={profile['mood']} | "
        f"energy={profile['energy']}"
    )
    return {
        "profile_key": profile_key,
        "profile_json": json.dumps(profile, sort_keys=True),
        "skip_recent": skip_recent,
        "query": query,
        "k": k,
        "use_llm_explainer": use_llm_explainer,
        "songs_path": songs_path,
        "run_clicked": run_clicked,
    }


def main() -> None:
    _inject_styles()
    _handle_spotify_callback()
    _render_header()
    controls = _render_sidebar()

    if controls["run_clicked"]:
        if not controls["query"].strip():
            st.warning("Enter a query before running the recommender.")
            return
        # Keep the submitted run so widget interactions (e.g. What If) rerender it
        # instead of resetting the page; results come from the cache.
        st.session_state["last_run"] = dict(controls)
    if "last_run" not in st.session_state:
        st.info("Choose a query and profile in the sidebar, then click Generate Recommendations.")
        return
    controls = st.session_state["last_run"]

    try:
        _load_catalog(controls["songs_path"])
        _warm_retriever(controls["songs_path"])
    except Exception as exc:
        st.error(f"Could not load the song catalog: {exc}")
        return

    args = (controls["query"], controls["profile_json"], controls["k"], controls["songs_path"],
            controls["skip_recent"])
    with st.spinner("Planning, retrieving, and ranking songs..."):
        try:
            result = _run_ranked(*args)
        except RuntimeError as exc:
            st.error(str(exc))
            st.stop()

    metrics_slot = st.empty()
    with metrics_slot.container():
        _render_metrics(result)
    _render_flag_legend(result)

    # Cards render immediately with templated narratives; the batched LLM explainer
    # then fills them in place.
    st.subheader("Recommendations")
    cards_slot = st.empty()
    with cards_slot.container():
        _render_recommendation_cards(result)

    if controls["use_llm_explainer"] and result.recommendations:
        with st.spinner("Writing explanations..."):
            result = _run_explained(*args)
        with cards_slot.container():
            _render_recommendation_cards(result)
        with metrics_slot.container():
            _render_metrics(result)

    st.subheader("Genre Mix")
    if result.recommendations:
        genre_counts = pd.Series([rec.song["genre"] for rec in result.recommendations]).value_counts()
        st.bar_chart(genre_counts)
    else:
        st.write("No chart available.")

    st.subheader("Run Summary")
    _render_status_summary(result)

    _render_what_if(result)
    _render_timings(result)
    _render_trace(result)


def _is_running_in_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    except ImportError:
        return False
    return get_script_run_ctx(suppress_warning=True) is not None


def _run_with_streamlit() -> None:
    from streamlit.web import cli as stcli

    script_path = str(Path(__file__).resolve())
    sys.argv = ["streamlit", "run", script_path, *sys.argv[1:]]
    raise SystemExit(stcli.main())


if __name__ == "__main__":
    if _is_running_in_streamlit():
        main()
    else:
        _run_with_streamlit()
