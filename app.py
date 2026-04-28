"""
Audio Annotation QA Simulator
Spotify Annotation QA Analyst Portfolio Project
Built by Nishtha Sood
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json
import io

from data_generator import generate_annotation_dataset, ANNOTATION_TASKS
from qa_engine import (
    compute_pairwise_kappa, compute_fleiss_kappa, flag_edge_cases,
    compute_category_stats, get_consensus_label, compute_annotator_stats
)
from report_generator import generate_qa_report_csv, generate_summary_stats

# ─── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Audio Annotation QA Simulator",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg: #0a0a0f;
    --surface: #111118;
    --surface2: #1a1a24;
    --accent: #1DB954;
    --accent2: #1ed760;
    --warn: #FF6B35;
    --danger: #E8163A;
    --text: #EAEAEA;
    --muted: #7a7a8a;
    --border: #2a2a38;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg);
    color: var(--text);
}

.stApp { background-color: var(--bg); }

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: var(--surface) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* Metrics */
[data-testid="stMetric"] {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px;
}
[data-testid="stMetricValue"] {
    font-family: 'Space Mono', monospace !important;
    color: var(--accent) !important;
    font-size: 2rem !important;
}
[data-testid="stMetricLabel"] { color: var(--muted) !important; }

/* Tabs */
[data-testid="stTabs"] button {
    font-family: 'DM Sans', sans-serif;
    font-weight: 600;
    color: var(--muted) !important;
    border-bottom: 2px solid transparent;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent);
}

/* Buttons */
.stButton > button {
    background: var(--accent) !important;
    color: #000 !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 50px !important;
    padding: 0.5rem 1.5rem !important;
    font-family: 'DM Sans', sans-serif !important;
    transition: all 0.2s ease;
}
.stButton > button:hover {
    background: var(--accent2) !important;
    transform: scale(1.02);
}

/* Dataframe */
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

/* Headers */
h1, h2, h3 { font-family: 'Space Mono', monospace !important; }
h1 { color: var(--accent) !important; }

/* Badges */
.badge-flag {
    display: inline-block;
    background: #E8163A22;
    color: #E8163A;
    border: 1px solid #E8163A55;
    border-radius: 20px;
    padding: 2px 12px;
    font-size: 12px;
    font-weight: 600;
    font-family: 'Space Mono', monospace;
}
.badge-ok {
    display: inline-block;
    background: #1DB95422;
    color: #1DB954;
    border: 1px solid #1DB95455;
    border-radius: 20px;
    padding: 2px 12px;
    font-size: 12px;
    font-weight: 600;
    font-family: 'Space Mono', monospace;
}
.badge-warn {
    display: inline-block;
    background: #FF6B3522;
    color: #FF6B35;
    border: 1px solid #FF6B3555;
    border-radius: 20px;
    padding: 2px 12px;
    font-size: 12px;
    font-weight: 600;
    font-family: 'Space Mono', monospace;
}

.kappa-card {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    margin-bottom: 10px;
}
.kappa-value {
    font-family: 'Space Mono', monospace;
    font-size: 2.5rem;
    font-weight: 700;
}
.kappa-label { color: var(--muted); font-size: 0.85rem; margin-top: 4px; }
.kappa-interp { font-size: 0.8rem; margin-top: 8px; font-weight: 600; }

.hero-header {
    background: linear-gradient(135deg, #0a0a0f 0%, #111118 50%, #0d1a10 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 32px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, #1DB95415 0%, transparent 70%);
    border-radius: 50%;
}

.guideline-box {
    background: var(--surface2);
    border-left: 3px solid var(--accent);
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    margin: 8px 0;
    font-size: 0.9rem;
}

select, [data-testid="stSelectbox"] div {
    background-color: var(--surface2) !important;
    color: var(--text) !important;
    border-color: var(--border) !important;
}
</style>
""", unsafe_allow_html=True)


# ─── SESSION STATE ──────────────────────────────────────────────────────────────
if "dataset" not in st.session_state:
    st.session_state.dataset = None
if "task" not in st.session_state:
    st.session_state.task = list(ANNOTATION_TASKS.keys())[0]
if "n_tracks" not in st.session_state:
    st.session_state.n_tracks = 200
if "n_annotators" not in st.session_state:
    st.session_state.n_annotators = 4
if "kappa_threshold" not in st.session_state:
    st.session_state.kappa_threshold = 0.60
if "disagreement_threshold" not in st.session_state:
    st.session_state.disagreement_threshold = 0.5


# ─── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎧 QA Simulator Config")
    st.markdown("---")

    task = st.selectbox(
        "Annotation Task",
        list(ANNOTATION_TASKS.keys()),
        help="Choose what annotators are labeling"
    )

    n_tracks = st.slider("Number of Tracks", 50, 500, 200, 25)
    n_annotators = st.slider("Annotators per Track", 3, 6, 4)
    kappa_threshold = st.slider("Kappa Alert Threshold", 0.2, 0.9, 0.60, 0.05,
                                 help="Flag categories below this Kappa score")
    disagreement_threshold = st.slider("Edge Case Threshold", 0.3, 0.8, 0.5, 0.05,
                                        help="Flag tracks with >X% disagreement rate")

    st.markdown("---")
    generate = st.button("⚡ Generate Dataset", use_container_width=True)

    st.markdown("---")
    st.markdown("**About this project**")
    st.markdown("""
    <div style='font-size:0.8rem; color:#7a7a8a; line-height:1.6'>
    Simulates a real annotation QA pipeline:<br>
    • Inter-annotator agreement (Cohen's κ, Fleiss' κ)<br>
    • Edge case detection & routing<br>
    • Ground truth consensus<br>
    • Annotator performance scoring<br>
    • QA reporting<br><br>
    Built by <b style='color:#1DB954'>Nishtha Sood</b>
    </div>
    """, unsafe_allow_html=True)


# ─── GENERATE / LOAD DATA ──────────────────────────────────────────────────────
if generate or st.session_state.dataset is None:
    with st.spinner("Generating annotation dataset..."):
        st.session_state.dataset = generate_annotation_dataset(
            task_name=task,
            n_tracks=n_tracks,
            n_annotators=n_annotators,
            seed=42
        )
        st.session_state.task = task
        st.session_state.n_tracks = n_tracks
        st.session_state.n_annotators = n_annotators
        st.session_state.kappa_threshold = kappa_threshold
        st.session_state.disagreement_threshold = disagreement_threshold

df = st.session_state.dataset
task_config = ANNOTATION_TASKS[st.session_state.task]
annotator_cols = [c for c in df.columns if c.startswith("annotator_")]


# ─── HERO HEADER ───────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero-header">
    <div style='display:flex; align-items:center; gap:16px; margin-bottom:8px'>
        <span style='font-size:2rem'>🎧</span>
        <h1 style='margin:0; font-size:1.8rem; font-family:Space Mono,monospace; color:#1DB954'>
            Audio Annotation QA Simulator
        </h1>
    </div>
    <div style='color:#7a7a8a; font-size:0.95rem; max-width:700px'>
        Replicating the annotation quality assurance pipeline used at ML/AI-driven audio platforms.
        Active task: <span style='color:#1DB954; font-weight:600'>{st.session_state.task}</span>
        &nbsp;·&nbsp; {st.session_state.n_tracks} tracks &nbsp;·&nbsp; {st.session_state.n_annotators} annotators
    </div>
</div>
""", unsafe_allow_html=True)


# ─── TOP-LINE METRICS ──────────────────────────────────────────────────────────
edge_cases = flag_edge_cases(df, annotator_cols, threshold=st.session_state.disagreement_threshold)
n_flagged = edge_cases["flagged"].sum()
n_clean = len(df) - n_flagged

fleiss_k = compute_fleiss_kappa(df, annotator_cols, task_config["labels"])
overall_agreement = (df[annotator_cols].apply(
    lambda row: (row == row.mode()[0]).mean(), axis=1
)).mean()

def kappa_color(k):
    if k >= 0.8: return "#1DB954"
    if k >= 0.6: return "#FFD700"
    if k >= 0.4: return "#FF6B35"
    return "#E8163A"

def kappa_interp(k):
    if k >= 0.8: return "Almost Perfect"
    if k >= 0.6: return "Substantial"
    if k >= 0.4: return "Moderate"
    if k >= 0.2: return "Fair"
    return "Slight / Poor"

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Tracks", f"{len(df):,}")
with col2:
    st.metric("Edge Cases Flagged", f"{n_flagged}", delta=f"{n_flagged/len(df)*100:.1f}% of dataset",
              delta_color="inverse")
with col3:
    st.metric("Overall Agreement", f"{overall_agreement:.1%}")
with col4:
    st.metric("Fleiss' Kappa (overall)", f"{fleiss_k:.3f}")

st.markdown("<br>", unsafe_allow_html=True)

# ─── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Agreement Analysis",
    "🚩 Edge Case Review",
    "👥 Annotator Performance",
    "📋 Annotation Guidelines",
    "📁 QA Report Export"
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: AGREEMENT ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Inter-Annotator Agreement (IAA) Dashboard")
    st.markdown("*Agreement metrics tell us how consistently annotators apply the same labels. Low agreement = ambiguous guidelines or hard edge cases.*")

    # Fleiss Kappa card
    col_k, col_info = st.columns([1, 2])
    with col_k:
        kc = kappa_color(fleiss_k)
        st.markdown(f"""
        <div class="kappa-card">
            <div class="kappa-value" style="color:{kc}">{fleiss_k:.3f}</div>
            <div class="kappa-label">Fleiss' Kappa — All Annotators</div>
            <div class="kappa-interp" style="color:{kc}">{kappa_interp(fleiss_k)}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_info:
        st.markdown("""
        **What is Fleiss' Kappa?**
        Fleiss' κ measures agreement among multiple annotators beyond chance.
        It's the gold standard metric in annotation QA pipelines.

        | κ Range | Interpretation |
        |---------|---------------|
        | 0.80–1.00 | Almost Perfect ✅ |
        | 0.60–0.79 | Substantial 🟡 |
        | 0.40–0.59 | Moderate 🟠 |
        | 0.20–0.39 | Fair 🔴 |
        | < 0.20 | Slight/Poor ❌ |
        """)

    st.markdown("---")

    # Pairwise Cohen's Kappa heatmap
    st.markdown("#### Pairwise Cohen's Kappa Matrix")
    st.caption("Agreement between each pair of annotators. Diagonal = 1.0. Off-diagonal shows where annotators diverge.")

    kappa_matrix = compute_pairwise_kappa(df, annotator_cols)
    annotator_labels = [f"Ann {i+1}" for i in range(len(annotator_cols))]

    fig_heatmap = go.Figure(data=go.Heatmap(
        z=kappa_matrix,
        x=annotator_labels,
        y=annotator_labels,
        colorscale=[
            [0.0, "#E8163A"], [0.4, "#FF6B35"],
            [0.6, "#FFD700"], [0.8, "#1DB954"], [1.0, "#1ed760"]
        ],
        zmin=0, zmax=1,
        text=[[f"{v:.3f}" for v in row] for row in kappa_matrix],
        texttemplate="%{text}",
        textfont={"size": 13, "family": "Space Mono"},
        hovertemplate="<b>%{y} vs %{x}</b><br>κ = %{z:.3f}<extra></extra>"
    ))
    fig_heatmap.update_layout(
        paper_bgcolor="#111118",
        plot_bgcolor="#111118",
        font=dict(color="#EAEAEA", family="DM Sans"),
        margin=dict(t=20, b=20, l=20, r=20),
        height=320,
        xaxis=dict(gridcolor="#2a2a38"),
        yaxis=dict(gridcolor="#2a2a38")
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)

    # Agreement distribution
    st.markdown("#### Track-Level Agreement Distribution")
    agreement_per_track = df[annotator_cols].apply(
        lambda row: (row == row.mode()[0]).mean(), axis=1
    )

    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(
        x=agreement_per_track,
        nbinsx=20,
        marker_color="#1DB954",
        marker_line_color="#0a0a0f",
        marker_line_width=1,
        opacity=0.85,
        name="Tracks"
    ))
    fig_dist.add_vline(
        x=st.session_state.disagreement_threshold,
        line_dash="dash", line_color="#E8163A", line_width=2,
        annotation_text=f"Edge Case Threshold ({st.session_state.disagreement_threshold:.0%})",
        annotation_font_color="#E8163A"
    )
    fig_dist.update_layout(
        paper_bgcolor="#111118", plot_bgcolor="#111118",
        font=dict(color="#EAEAEA", family="DM Sans"),
        xaxis=dict(title="Agreement Rate", gridcolor="#2a2a38", tickformat=".0%"),
        yaxis=dict(title="Number of Tracks", gridcolor="#2a2a38"),
        margin=dict(t=20, b=40, l=40, r=20),
        height=280, showlegend=False
    )
    st.plotly_chart(fig_dist, use_container_width=True)

    # Category-level breakdown
    st.markdown("#### Agreement by Label Category")
    cat_stats = compute_category_stats(df, annotator_cols, task_config["labels"])
    cat_df = pd.DataFrame(cat_stats).T
    cat_df.index.name = "Label"
    cat_df = cat_df.reset_index()

    fig_cat = go.Figure()
    colors = ["#1DB954" if v >= st.session_state.kappa_threshold else "#E8163A"
              for v in cat_df["avg_agreement"]]
    fig_cat.add_trace(go.Bar(
        x=cat_df["Label"],
        y=cat_df["avg_agreement"],
        marker_color=colors,
        text=[f"{v:.1%}" for v in cat_df["avg_agreement"]],
        textposition="outside",
        textfont=dict(family="Space Mono", size=11),
    ))
    fig_cat.add_hline(
        y=st.session_state.kappa_threshold,
        line_dash="dot", line_color="#FFD700",
        annotation_text=f"QA threshold ({st.session_state.kappa_threshold:.0%})",
        annotation_font_color="#FFD700"
    )
    fig_cat.update_layout(
        paper_bgcolor="#111118", plot_bgcolor="#111118",
        font=dict(color="#EAEAEA", family="DM Sans"),
        xaxis=dict(gridcolor="#2a2a38"),
        yaxis=dict(title="Avg Agreement", gridcolor="#2a2a38", tickformat=".0%", range=[0, 1.1]),
        margin=dict(t=40, b=40, l=40, r=20),
        height=300, showlegend=False
    )
    st.plotly_chart(fig_cat, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: EDGE CASE REVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Edge Case Detection & Review Queue")
    st.markdown("*Tracks where annotators significantly disagree — these require human QA review to establish ground truth.*")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        pct = n_flagged / len(df) * 100
        badge = "badge-flag" if pct > 20 else "badge-warn" if pct > 10 else "badge-ok"
        st.markdown(f"""
        <div class="kappa-card">
            <div class="kappa-value" style="color:#E8163A">{n_flagged}</div>
            <div class="kappa-label">Tracks Flagged for Review</div>
            <div class='kappa-interp'><span class='{badge}'>{pct:.1f}% of dataset</span></div>
        </div>
        """, unsafe_allow_html=True)
    with col_b:
        st.markdown(f"""
        <div class="kappa-card">
            <div class="kappa-value" style="color:#1DB954">{n_clean}</div>
            <div class="kappa-label">Tracks Passed QA</div>
            <div class='kappa-interp'><span class='badge-ok'>consensus reached</span></div>
        </div>
        """, unsafe_allow_html=True)
    with col_c:
        avg_conf = edge_cases[edge_cases["flagged"]]["disagreement_rate"].mean() if n_flagged > 0 else 0
        st.markdown(f"""
        <div class="kappa-card">
            <div class="kappa-value" style="color:#FF6B35">{avg_conf:.1%}</div>
            <div class="kappa-label">Avg Disagreement (Flagged)</div>
            <div class='kappa-interp'><span class='badge-warn'>needs review</span></div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    flagged_df = edge_cases[edge_cases["flagged"]].copy()

    if len(flagged_df) == 0:
        st.success("✅ No edge cases detected at current threshold. Try lowering the Edge Case Threshold in the sidebar.")
    else:
        # Add consensus column
        flagged_df["consensus"] = flagged_df[annotator_cols].apply(
            lambda row: get_consensus_label(row.tolist()), axis=1
        )

        # Severity buckets
        flagged_df["severity"] = flagged_df["disagreement_rate"].apply(
            lambda x: "🔴 HIGH" if x >= 0.7 else "🟠 MEDIUM" if x >= 0.5 else "🟡 LOW"
        )

        # Interactive review interface
        st.markdown("#### Review Queue")
        st.caption(f"Showing {len(flagged_df)} flagged tracks ordered by disagreement severity")

        display_cols = ["track_name", "artist", "genre"] + annotator_cols + ["disagreement_rate", "consensus", "severity"]
        display_df = flagged_df[display_cols].rename(columns={
            "track_name": "Track",
            "artist": "Artist",
            "genre": "Genre",
            "disagreement_rate": "Disagree %",
            "consensus": "Suggested GT",
            "severity": "Severity"
        }).sort_values("Disagree %", ascending=False)

        # Format disagreement
        display_df["Disagree %"] = display_df["Disagree %"].apply(lambda x: f"{x:.1%}")

        st.dataframe(
            display_df,
            use_container_width=True,
            height=400,
            column_config={
                "Track": st.column_config.TextColumn("Track", width="medium"),
                "Severity": st.column_config.TextColumn("Severity", width="small"),
                "Suggested GT": st.column_config.TextColumn("Suggested GT", width="small"),
            }
        )

        # Disagreement by genre
        st.markdown("#### Edge Cases by Genre")
        genre_edge = flagged_df.groupby("genre").size().reset_index(name="flagged_count")
        genre_total = df.groupby("genre").size().reset_index(name="total")
        genre_merged = genre_edge.merge(genre_total, on="genre")
        genre_merged["flag_rate"] = genre_merged["flagged_count"] / genre_merged["total"]
        genre_merged = genre_merged.sort_values("flag_rate", ascending=True)

        fig_genre = go.Figure(go.Bar(
            x=genre_merged["flag_rate"],
            y=genre_merged["genre"],
            orientation='h',
            marker_color="#FF6B35",
            text=[f"{v:.0%}" for v in genre_merged["flag_rate"]],
            textposition="outside"
        ))
        fig_genre.update_layout(
            paper_bgcolor="#111118", plot_bgcolor="#111118",
            font=dict(color="#EAEAEA", family="DM Sans"),
            xaxis=dict(title="Edge Case Rate", gridcolor="#2a2a38", tickformat=".0%"),
            yaxis=dict(gridcolor="#2a2a38"),
            margin=dict(t=20, b=40, l=120, r=60),
            height=300, showlegend=False
        )
        st.plotly_chart(fig_genre, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: ANNOTATOR PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### Annotator Performance Scoring")
    st.markdown("*Track individual annotator consistency, agreement with consensus, and contribution to edge cases.*")

    ann_stats = compute_annotator_stats(df, annotator_cols)

    # Radar chart
    categories = ["Consistency", "Agreement Rate", "Edge Case Rate (inv)", "Confidence Score"]
    fig_radar = go.Figure()
    colors_r = ["#1DB954", "#1ed760", "#17a349", "#FFD700", "#FF6B35", "#E8163A"]

    for i, (ann_id, stats) in enumerate(ann_stats.items()):
        fig_radar.add_trace(go.Scatterpolar(
            r=[
                stats["consistency"],
                stats["agreement_rate"],
                1 - stats["edge_case_contribution"],
                stats["confidence"]
            ],
            theta=categories,
            fill='toself',
            name=f"Annotator {i+1}",
            line_color=colors_r[i % len(colors_r)],
            fillcolor=colors_r[i % len(colors_r)],
            opacity=0.3
        ))
    fig_radar.update_layout(
        polar=dict(
            bgcolor="#111118",
            radialaxis=dict(visible=True, range=[0, 1], gridcolor="#2a2a38", color="#7a7a8a"),
            angularaxis=dict(gridcolor="#2a2a38", color="#EAEAEA")
        ),
        paper_bgcolor="#111118",
        font=dict(color="#EAEAEA", family="DM Sans"),
        legend=dict(bgcolor="#1a1a24", bordercolor="#2a2a38"),
        height=400,
        margin=dict(t=30, b=30, l=60, r=60)
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # Annotator table
    ann_table = []
    for i, (ann_id, stats) in enumerate(ann_stats.items()):
        score = (stats["consistency"] * 0.4 + stats["agreement_rate"] * 0.4 + stats["confidence"] * 0.2)
        status = "✅ Strong" if score >= 0.75 else "🟡 Review" if score >= 0.60 else "🔴 Flag"
        ann_table.append({
            "Annotator": f"Annotator {i+1}",
            "Consistency": f"{stats['consistency']:.1%}",
            "Agreement Rate": f"{stats['agreement_rate']:.1%}",
            "Edge Case Contrib.": f"{stats['edge_case_contribution']:.1%}",
            "Confidence Score": f"{stats['confidence']:.2f}",
            "QA Score": f"{score:.1%}",
            "Status": status
        })

    st.dataframe(pd.DataFrame(ann_table), use_container_width=True, hide_index=True)

    st.info("💡 **QA Insight:** Annotators with high edge case contribution may need additional guideline training. Consider calibration sessions for pairs with low pairwise Kappa scores.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: ANNOTATION GUIDELINES
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### Annotation Guidelines & Ground Truth Framework")
    st.markdown("*Living guidelines that define how annotators should label audio content — the foundation of annotation consistency.*")

    task_cfg = ANNOTATION_TASKS[st.session_state.task]

    col_g1, col_g2 = st.columns([1, 1])
    with col_g1:
        st.markdown(f"#### Task: {st.session_state.task}")
        st.markdown(f"**Description:** {task_cfg['description']}")
        st.markdown(f"**Label Space:** `{'` · `'.join(task_cfg['labels'])}`")
        st.markdown(f"**Modalities:** Audio features, metadata, lyrics context")

        st.markdown("#### Decision Rules")
        for rule in task_cfg["guidelines"]:
            st.markdown(f"""<div class="guideline-box">{rule}</div>""", unsafe_allow_html=True)

    with col_g2:
        st.markdown("#### Edge Case Handling Protocol")
        st.markdown("""
        <div class="guideline-box">🔴 <b>Step 1:</b> Flag tracks where annotator disagreement rate exceeds threshold</div>
        <div class="guideline-box">🟠 <b>Step 2:</b> Route to QA Review Queue for senior annotator review</div>
        <div class="guideline-box">🟡 <b>Step 3:</b> Apply consensus voting — majority label becomes ground truth</div>
        <div class="guideline-box">🟢 <b>Step 4:</b> If no majority, escalate to Content Policy team</div>
        <div class="guideline-box">✅ <b>Step 5:</b> Update guidelines to reduce future ambiguity</div>
        """, unsafe_allow_html=True)

        st.markdown("#### Ground Truth Confidence Levels")
        conf_data = {
            "Confidence": ["High (≥80% agree)", "Medium (60–79%)", "Low (<60%)"],
            "Action": ["Accept as GT", "Flag for review", "Escalate to policy"],
            "Volume": [f"{n_clean}", f"{n_flagged//2}", f"{n_flagged - n_flagged//2}"]
        }
        st.dataframe(pd.DataFrame(conf_data), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("#### Label Distribution in Dataset")
    all_labels = df[annotator_cols].values.flatten()
    label_counts = pd.Series(all_labels).value_counts().reset_index()
    label_counts.columns = ["Label", "Count"]

    fig_labels = px.pie(
        label_counts, values="Count", names="Label",
        color_discrete_sequence=["#1DB954", "#FFD700", "#FF6B35", "#E8163A", "#9B59B6", "#3498DB"],
        hole=0.5
    )
    fig_labels.update_layout(
        paper_bgcolor="#111118",
        font=dict(color="#EAEAEA", family="DM Sans"),
        legend=dict(bgcolor="#1a1a24", bordercolor="#2a2a38"),
        margin=dict(t=20, b=20, l=20, r=20),
        height=300
    )
    st.plotly_chart(fig_labels, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5: EXPORT
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("### QA Report Export")
    st.markdown("*Generate and download structured QA reports for handoff to Product, Engineering, and Content Policy teams.*")

    summary = generate_summary_stats(df, annotator_cols, edge_cases, fleiss_k, task_config)

    col_e1, col_e2 = st.columns(2)

    with col_e1:
        st.markdown("#### Summary Report")
        st.json(summary)

    with col_e2:
        st.markdown("#### Export Options")

        # Full dataset CSV
        csv_full = generate_qa_report_csv(df, annotator_cols, edge_cases)
        st.download_button(
            "⬇️ Download Full QA Dataset (CSV)",
            data=csv_full,
            file_name=f"annotation_qa_full_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )

        # Edge cases only
        edge_csv = generate_qa_report_csv(
            edge_cases[edge_cases["flagged"]],
            annotator_cols,
            edge_cases[edge_cases["flagged"]]
        )
        st.download_button(
            "🚩 Download Edge Cases Only (CSV)",
            data=edge_csv,
            file_name=f"edge_cases_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )

        # Summary JSON
        st.download_button(
            "📋 Download QA Summary (JSON)",
            data=json.dumps(summary, indent=2),
            file_name=f"qa_summary_{datetime.now().strftime('%Y%m%d')}.json",
            mime="application/json",
            use_container_width=True
        )

    st.markdown("---")
    st.markdown("#### Report Preview — Flagged Items")
    preview_df = edge_cases[edge_cases["flagged"]][
        ["track_name", "artist", "genre"] + annotator_cols + ["disagreement_rate"]
    ].head(20)
    preview_df["disagreement_rate"] = preview_df["disagreement_rate"].apply(lambda x: f"{x:.1%}")
    st.dataframe(preview_df, use_container_width=True, hide_index=True)
