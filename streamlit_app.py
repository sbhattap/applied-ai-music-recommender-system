"""Streamlit UI for the agent-orchestrated music recommender.
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from src.recommender import config
from src.recommender.core.io import load_songs
from src.recommender.llm import agent
from src.recommender.main import PROFILES


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
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def _load_catalog(csv_path: str) -> List[Dict[str, Any]]:
    return load_songs(csv_path)


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


def _profile_label(profile_key: str) -> str:
    profile = PROFILES[profile_key]
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
    metric_cols = st.columns(4)
    metric_cols[0].metric("Candidates", result.candidate_count)
    metric_cols[1].metric("Recommendations", len(result.recommendations))
    metric_cols[2].metric("Confidence", f"{result.verdict.confidence:.2f}")
    metric_cols[3].metric("Flags", len(flags))


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


def _render_recommendation_cards(result: agent.AgentResult) -> None:
    if not result.recommendations:
        st.warning("No recommendations were returned for this query.")
        return

    cols = st.columns(2)
    for index, rec in enumerate(result.recommendations):
        song = rec.song
        with cols[index % 2]:
            st.markdown(
                f"""
                <div class="rec-card">
                  <div class="rec-kicker">Recommendation {index + 1}</div>
                  <div class="rec-title">{song['title']}</div>
                  <div class="rec-meta">{song['artist']}</div>
                  <div class="pill-row">
                    <span class="pill">{song['genre']}</span>
                    <span class="pill">{song['mood']}</span>
                    <span class="pill">score {rec.score:.3f}</span>
                    <span class="pill">instrumental {song.get('instrumentalness', 0.0):.2f}</span>
                  </div>
                  <div class="rec-body">{rec.narrative}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            with st.expander("Why this matched"):
                st.write(rec.signal_explanation)


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
    profile_key = st.sidebar.selectbox(
        "Profile",
        options=list(PROFILES.keys()),
        format_func=_profile_label,
        index=list(PROFILES.keys()).index("chill_lofi"),
    )
    query = st.sidebar.text_area(
        "Query",
        value="chill study music, no vocals",
        height=110,
    )
    k = st.sidebar.slider("Recommendations", min_value=1, max_value=10, value=5)
    use_llm_explainer = st.sidebar.checkbox("Use LLM explainer", value=True)
    songs_path = st.sidebar.text_input("Songs CSV", value=str(config.SONGS_CSV))
    run_clicked = st.sidebar.button("Generate Recommendations", use_container_width=True, type="primary")

    st.sidebar.markdown("---")
    profile = PROFILES[profile_key]
    st.sidebar.caption(
        f"Profile: {profile['name']} | genre={profile['genre']} | mood={profile['mood']} | "
        f"energy={profile['energy']}"
    )
    return {
        "profile_key": profile_key,
        "query": query,
        "k": k,
        "use_llm_explainer": use_llm_explainer,
        "songs_path": songs_path,
        "run_clicked": run_clicked,
    }


def main() -> None:
    _inject_styles()
    _render_header()
    controls = _render_sidebar()

    if not controls["run_clicked"]:
        st.info("Choose a query and profile in the sidebar, then click Generate Recommendations.")
        return

    if not controls["query"].strip():
        st.warning("Enter a query before running the recommender.")
        return

    try:
        songs = _load_catalog(controls["songs_path"])
    except Exception as exc:
        st.error(f"Could not load the song catalog: {exc}")
        return

    profile = PROFILES[controls["profile_key"]]
    with st.spinner("Planning, retrieving, and ranking songs..."):
        try:
            result = agent.run(
                controls["query"],
                profile,
                songs,
                k=controls["k"],
                use_llm_explainer=controls["use_llm_explainer"],
            )
        except RuntimeError as exc:
            st.error(str(exc))
            st.stop()

    _render_metrics(result)
    _render_flag_legend(result)

    st.subheader("Recommendations")
    _render_recommendation_cards(result)

    st.subheader("Genre Mix")
    if result.recommendations:
        genre_counts = pd.Series([rec.song["genre"] for rec in result.recommendations]).value_counts()
        st.bar_chart(genre_counts)
    else:
        st.write("No chart available.")

    st.subheader("Run Summary")
    _render_status_summary(result)

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
