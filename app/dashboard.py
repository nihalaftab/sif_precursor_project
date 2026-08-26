"""
SIF Precursor Detection — Interactive Streamlit Dashboard
Oil India Limited | HSSE AI/NLP Engine

5-page dashboard:
  1. Upload & Analyze
  2. SIF Risk Heatmap
  3. Life-Saving Rule Dashboard
  4. Precursor Pattern Explorer
  5. Report Explorer
"""

import sys
import os
# Ensure project root is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import io

from utils.config import (
    PAGE_TITLE, PAGE_ICON, LIFE_SAVING_RULES, SIF_SCORE_THRESHOLD,
)
from utils.exporter import to_excel_bytes, to_pdf_bytes

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #f0f4f8; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a3a5c 0%, #0d2238 100%);
        color: white;
    }
    [data-testid="stSidebar"] * { color: white !important; }
    [data-testid="stSidebar"] .stSelectbox label { color: white !important; }

    /* Metric cards */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 20px 24px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 5px solid #1a3a5c;
        margin-bottom: 12px;
    }
    .metric-card.danger { border-left-color: #e74c3c; }
    .metric-card.warning { border-left-color: #f39c12; }
    .metric-card.success { border-left-color: #27ae60; }
    .metric-value { font-size: 2.5rem; font-weight: 700; color: #1a3a5c; }
    .metric-label { font-size: 0.85rem; color: #666; text-transform: uppercase; letter-spacing: 0.5px; }

    /* SIF badge */
    .sif-badge-danger {
        background: #e74c3c; color: white; padding: 3px 10px;
        border-radius: 12px; font-weight: 600; font-size: 0.8rem;
    }
    .sif-badge-ok {
        background: #27ae60; color: white; padding: 3px 10px;
        border-radius: 12px; font-weight: 600; font-size: 0.8rem;
    }

    /* Section headers */
    .section-header {
        font-size: 1.4rem; font-weight: 700; color: #1a3a5c;
        border-bottom: 3px solid #e74c3c; padding-bottom: 8px;
        margin-bottom: 20px;
    }

    /* Report card */
    .report-card {
        background: white; border-radius: 10px;
        padding: 16px; margin-bottom: 10px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }

    /* Top bar */
    .top-bar {
        background: linear-gradient(90deg, #1a3a5c, #c0392b);
        color: white; padding: 16px 24px; border-radius: 10px;
        margin-bottom: 20px;
    }
    .top-bar h1 { color: white !important; margin: 0; font-size: 1.6rem; }
    .top-bar p { color: rgba(255,255,255,0.85); margin: 4px 0 0; }
</style>
""", unsafe_allow_html=True)


# ── Session State ─────────────────────────────────────────────────────────────
if "results_df" not in st.session_state:
    st.session_state.results_df = None
if "patterns" not in st.session_state:
    st.session_state.patterns = None


# ── Sidebar Navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🛢️ OIL — SIF Precursor Engine")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        [
            "📤 Upload & Analyze",
            "🔴 SIF Risk Heatmap",
            "🏷️ Life-Saving Rules",
            "🔍 Pattern Explorer",
            "📋 Report Explorer",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("**Threshold Settings**")
    sif_threshold = st.slider(
        "SIF Score Threshold", 0.1, 0.9, SIF_SCORE_THRESHOLD, 0.05,
        help="Reports with SIF score above this are flagged as SIF-potential"
    )
    use_llm = st.checkbox(
        "🤖 Use Zero-Shot LLM", value=False,
        help="Enable bart-large-mnli for more accurate classification (slower)"
    )
    st.markdown("---")
    st.caption("v1.0 | Oil India Limited HSSE")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — UPLOAD & ANALYZE
# ═══════════════════════════════════════════════════════════════════════════════
def page_upload():
    st.markdown("""
    <div class="top-bar">
        <h1>🛢️ SIF Precursor Detection Engine</h1>
        <p>AI/NLP engine to classify Serious Injury & Fatality precursors in OIL's safety reports</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<div class="section-header">📤 Upload Safety Reports</div>', unsafe_allow_html=True)
        st.markdown("""
        Upload your UA/UC, Near-Miss, or Incident reports in CSV format.
        Required column: **`narrative`** (free-text report description).
        Optional columns: `report_id`, `date`, `site`, `department`, `report_type`
        """)

        uploaded = st.file_uploader(
            "Upload CSV file", type=["csv"],
            help="CSV with at minimum a 'narrative' column"
        )

        use_demo = st.button("🔬 Use Demo Dataset (500 Synthetic OIL Reports)", use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">⚙️ Processing Pipeline</div>', unsafe_allow_html=True)
        st.markdown("""
        The engine runs the following steps on each report:

        | Step | Module | Output |
        |---|---|---|
        | 1️⃣ Preprocess | Abbreviation expansion + NER | Clean text, entities |
        | 2️⃣ Classify | Keyword + LLM ensemble | SIF score, confidence |
        | 3️⃣ Tag | IOGP LSR semantic matcher | Life-Saving Rule |
        | 4️⃣ Mine | BERTopic cluster analysis | Precursor patterns |
        """)

    # ── Data Loading ──────────────────────────────────────────────────────────
    df_raw = None
    if use_demo:
        with st.spinner("Generating demo dataset..."):
            sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
            from data.synthetic_reports import generate_dataset
            df_raw = generate_dataset(500)
            st.success("✅ Demo dataset loaded — 500 synthetic OIL safety reports")

    elif uploaded:
        try:
            df_raw = pd.read_csv(uploaded)
            st.success(f"✅ Uploaded {len(df_raw)} reports from {uploaded.name}")
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            return

    if df_raw is None:
        _show_sample_output()
        return

    # ── Validate columns ──────────────────────────────────────────────────────
    if "narrative" not in df_raw.columns:
        st.error("❌ CSV must contain a 'narrative' column with report text.")
        return

    # Fill optional columns
    for col, default in [
        ("report_id",   [f"RPT-{i+1:04d}" for i in range(len(df_raw))]),
        ("date",        [datetime.now().strftime("%Y-%m-%d")] * len(df_raw)),
        ("site",        ["Unknown Site"] * len(df_raw)),
        ("department",  ["Unknown Dept"] * len(df_raw)),
        ("report_type", ["Observation"] * len(df_raw)),
    ]:
        if col not in df_raw.columns:
            df_raw[col] = default

    # ── Run Pipeline ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-header">🚀 Running Analysis Pipeline</div>', unsafe_allow_html=True)

    progress_bar = st.progress(0, text="Initialising NLP engine...")
    status_text  = st.empty()

    try:
        from engine.preprocessor   import preprocess, get_embeddings_batch
        from engine.sif_classifier  import classify_report
        from engine.lsr_tagger      import tag_report
        from engine.pattern_miner   import mine_patterns, summarise_patterns

        narratives = df_raw["narrative"].fillna("").tolist()
        n = len(narratives)

        # Step 1: Preprocess
        status_text.text("Step 1/4 — Preprocessing text & extracting entities...")
        preprocessed = []
        for i, narrative in enumerate(narratives):
            preprocessed.append(preprocess(narrative))
            if (i + 1) % 10 == 0:
                progress_bar.progress((i + 1) / (n * 4), text=f"Preprocessing {i+1}/{n}...")

        clean_texts = [p["clean_text"] for p in preprocessed]

        # Step 2: Compute embeddings (batch)
        progress_bar.progress(0.25, text="Step 2/4 — Computing sentence embeddings...")
        status_text.text("Step 2/4 — Computing sentence embeddings (batch)...")
        embeddings = get_embeddings_batch(clean_texts, show_progress=False)

        # Step 3: Classify + Tag
        progress_bar.progress(0.50, text="Step 3/4 — Classifying SIF potential & tagging LSR...")
        status_text.text("Step 3/4 — Classifying SIF potential & tagging Life-Saving Rules...")
        clf_results = []
        lsr_results = []
        for i, (text, emb) in enumerate(zip(clean_texts, embeddings)):
            clf_results.append(classify_report(text, use_llm=use_llm))
            lsr_results.append(tag_report(text, text_embedding=emb))
            if (i + 1) % 10 == 0:
                progress_bar.progress(0.50 + 0.30 * (i + 1) / n, text=f"Classifying {i+1}/{n}...")

        # Step 4: Pattern Mining (SIF-only)
        progress_bar.progress(0.80, text="Step 4/4 — Mining precursor patterns...")
        status_text.text("Step 4/4 — Mining recurring precursor patterns (BERTopic)...")
        sif_mask   = [r["sif_potential"] for r in clf_results]
        sif_texts  = [t for t, m in zip(clean_texts, sif_mask) if m]
        sif_embs   = embeddings[[i for i, m in enumerate(sif_mask) if m]] if any(sif_mask) else None
        patterns   = mine_patterns(sif_texts, sif_embs, n_topics=10)
        st.session_state.patterns = patterns

        # Assemble results DataFrame
        results = []
        for i, row in df_raw.iterrows():
            clf = clf_results[i]
            lsr = lsr_results[i]
            pre = preprocessed[i]
            results.append({
                "report_id":       row["report_id"],
                "date":            row["date"],
                "site":            row["site"],
                "department":      row["department"],
                "report_type":     row["report_type"],
                "narrative":       row["narrative"],
                "sif_potential":   clf["sif_potential"],
                "sif_score":       clf["sif_score"],
                "confidence":      clf["confidence"],
                "life_saving_rule": lsr["primary_rule"],
                "all_lsr_tags":    ", ".join(lsr["tagged_rules"]),
                "top_signals":     ", ".join(clf["top_signals"]),
                "explanation":     clf["explanation"],
                "entities_equipment": ", ".join(pre["entities"]["equipment"][:3]),
                "entities_location":  ", ".join(pre["entities"]["location"][:3]),
                "entities_activity":  ", ".join(pre["entities"]["activity"][:3]),
            })

        results_df = pd.DataFrame(results)
        st.session_state.results_df = results_df

        progress_bar.progress(1.0, text="✅ Analysis complete!")
        status_text.empty()

        # ── Summary KPIs ───────────────────────────────────────────────────────
        total    = len(results_df)
        sif_cnt  = results_df["sif_potential"].sum()
        sif_pct  = round(100 * sif_cnt / total, 1) if total else 0
        high_cnt = (results_df["confidence"] == "HIGH").sum()
        top_site = (
            results_df[results_df["sif_potential"]]["site"]
            .value_counts().index[0]
            if sif_cnt > 0 else "N/A"
        )

        st.markdown("---")
        st.markdown('<div class="section-header">📊 Summary Results</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("📄 Total Reports", total)
        with c2:
            st.metric("🔴 SIF-Potential", f"{sif_cnt} ({sif_pct}%)", delta=f"+{sif_cnt}")
        with c3:
            st.metric("⚠️ HIGH Confidence", high_cnt)
        with c4:
            st.metric("📍 Highest-Risk Site", top_site)

        # SIF vs Non-SIF donut
        fig_donut = px.pie(
            names=["SIF-Potential", "Non-SIF"],
            values=[sif_cnt, total - sif_cnt],
            hole=0.55,
            color_discrete_sequence=["#e74c3c", "#27ae60"],
            title="SIF vs Non-SIF Classification Split",
        )
        fig_donut.update_traces(textposition="outside", textinfo="percent+label")
        fig_donut.update_layout(height=320, margin=dict(t=40, b=10))

        # Confidence distribution
        conf_counts = results_df[results_df["sif_potential"]]["confidence"].value_counts()
        fig_conf = px.bar(
            x=conf_counts.index, y=conf_counts.values,
            color=conf_counts.index,
            color_discrete_map={"HIGH": "#e74c3c", "MEDIUM": "#f39c12", "LOW": "#95a5a6"},
            title="SIF Confidence Level Distribution",
            labels={"x": "Confidence", "y": "Count"},
        )
        fig_conf.update_layout(height=320, showlegend=False, margin=dict(t=40, b=10))

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_donut, use_container_width=True)
        with col2:
            st.plotly_chart(fig_conf, use_container_width=True)

        # Export buttons
        st.markdown("---")
        col_dl1, col_dl2, _ = st.columns([1, 1, 2])
        with col_dl1:
            excel_bytes = to_excel_bytes(results_df)
            st.download_button(
                "📥 Download Excel",
                data=excel_bytes,
                file_name=f"sif_results_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        with col_dl2:
            pdf_bytes = to_pdf_bytes(results_df)
            if pdf_bytes:
                st.download_button(
                    "📄 Download PDF",
                    data=pdf_bytes,
                    file_name=f"sif_results_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )

        st.success("✅ Analysis complete! Navigate to other pages using the sidebar.")

    except Exception as e:
        progress_bar.progress(0)
        st.error(f"❌ Pipeline error: {e}")
        st.exception(e)


def _show_sample_output():
    """Show sample output when no data is loaded."""
    st.markdown("---")
    st.markdown('<div class="section-header">💡 Sample Output Preview</div>', unsafe_allow_html=True)
    sample = pd.DataFrame([
        {
            "report_id": "RPT-2024-0387",
            "narrative": "Worker entered confined tank without atmospheric test. H2S levels were later found at 45 ppm.",
            "sif_potential": "✅ YES",
            "sif_score": 0.94,
            "confidence": "HIGH",
            "life_saving_rule": "Confined Space",
            "top_signals": "h2s, atmospheric test, confined tank",
        },
        {
            "report_id": "RPT-2024-0201",
            "narrative": "Hard hat tilted at angle. Worker reminded and corrected.",
            "sif_potential": "❌ NO",
            "sif_score": 0.05,
            "confidence": "LOW",
            "life_saving_rule": "—",
            "top_signals": "—",
        },
    ])
    st.dataframe(sample, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — SIF RISK HEATMAP
# ═══════════════════════════════════════════════════════════════════════════════
def page_heatmap():
    st.markdown('<div class="section-header">🔴 SIF Risk Heatmap — Site-Level Risk Ranking</div>', unsafe_allow_html=True)
    df = _get_results()
    if df is None:
        return

    # ── Filters ───────────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    with col1:
        all_sites = ["All Sites"] + sorted(df["site"].unique().tolist())
        site_filter = st.selectbox("Filter by Site", all_sites)
    with col2:
        all_types = ["All Types"] + sorted(df["report_type"].unique().tolist())
        type_filter = st.selectbox("Filter by Report Type", all_types)
    with col3:
        all_depts = ["All Departments"] + sorted(df["department"].unique().tolist())
        dept_filter = st.selectbox("Filter by Department", all_depts)

    filtered = df.copy()
    if site_filter != "All Sites":
        filtered = filtered[filtered["site"] == site_filter]
    if type_filter != "All Types":
        filtered = filtered[filtered["report_type"] == type_filter]
    if dept_filter != "All Departments":
        filtered = filtered[filtered["department"] == dept_filter]

    # ── Site Risk Summary Table ────────────────────────────────────────────────
    site_summary = (
        filtered.groupby("site")
        .agg(
            total_reports=("report_id", "count"),
            sif_count=("sif_potential", "sum"),
            avg_sif_score=("sif_score", "mean"),
        )
        .reset_index()
    )
    site_summary["sif_pct"] = (100 * site_summary["sif_count"] / site_summary["total_reports"]).round(1)
    site_summary["risk_level"] = site_summary["sif_pct"].apply(
        lambda x: "🔴 HIGH" if x >= 40 else ("🟡 MEDIUM" if x >= 20 else "🟢 LOW")
    )
    site_summary = site_summary.sort_values("sif_pct", ascending=False)

    # ── Horizontal Bar Chart — SIF % by Site ─────────────────────────────────
    colors_map = site_summary["sif_pct"].apply(
        lambda x: "#e74c3c" if x >= 40 else ("#f39c12" if x >= 20 else "#27ae60")
    )
    fig_bar = go.Figure(go.Bar(
        x=site_summary["sif_pct"],
        y=site_summary["site"],
        orientation="h",
        marker_color=colors_map,
        text=site_summary["sif_pct"].apply(lambda x: f"{x}%"),
        textposition="outside",
        hovertemplate=(
            "<b>%{y}</b><br>"
            "SIF Density: %{x}%<br>"
            "<extra></extra>"
        ),
    ))
    fig_bar.update_layout(
        title="SIF-Precursor Density by Site (%)",
        xaxis_title="SIF-Potential Reports (%)",
        height=400,
        margin=dict(l=10, r=80, t=40, b=10),
        plot_bgcolor="#fafafa",
        xaxis=dict(range=[0, max(site_summary["sif_pct"].max() * 1.2, 10)]),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # ── Bubble Chart — Volume vs Severity ────────────────────────────────────
    fig_bubble = px.scatter(
        site_summary,
        x="total_reports",
        y="sif_pct",
        size="sif_count",
        color="sif_pct",
        text="site",
        color_continuous_scale="RdYlGn_r",
        title="Site Risk Matrix: Volume vs SIF Density",
        labels={
            "total_reports": "Total Reports",
            "sif_pct": "SIF Density (%)",
            "sif_count": "SIF Count",
        },
        size_max=50,
    )
    fig_bubble.update_traces(textposition="top center")
    fig_bubble.update_layout(height=450, coloraxis_showscale=True)
    st.plotly_chart(fig_bubble, use_container_width=True)

    # ── Heatmap: Site × Department ────────────────────────────────────────────
    st.markdown("##### SIF Count Heatmap: Site × Department")
    pivot = filtered[filtered["sif_potential"]].pivot_table(
        index="site", columns="department", values="report_id", aggfunc="count", fill_value=0
    )
    if not pivot.empty:
        fig_heat = px.imshow(
            pivot,
            color_continuous_scale="Reds",
            title="SIF Reports Heatmap (Site × Department)",
            text_auto=True,
        )
        fig_heat.update_layout(height=400)
        st.plotly_chart(fig_heat, use_container_width=True)

    # ── Risk Table ────────────────────────────────────────────────────────────
    st.markdown("##### 📊 Site Risk Ranking Table")
    display_cols = ["site", "total_reports", "sif_count", "sif_pct", "avg_sif_score", "risk_level"]
    display_df = site_summary[display_cols].rename(columns={
        "site": "Site", "total_reports": "Total Reports",
        "sif_count": "SIF Reports", "sif_pct": "SIF %",
        "avg_sif_score": "Avg SIF Score", "risk_level": "Risk Level"
    })
    display_df["Avg SIF Score"] = display_df["Avg SIF Score"].round(3)
    st.dataframe(display_df, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — LIFE-SAVING RULES DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
def page_lsr():
    st.markdown('<div class="section-header">🏷️ IOGP Life-Saving Rule Dashboard</div>', unsafe_allow_html=True)
    df = _get_results()
    if df is None:
        return

    sif_df = df[df["sif_potential"]].copy()

    if sif_df.empty:
        st.warning("No SIF-potential reports found in the dataset.")
        return

    # ── Top KPIs ──────────────────────────────────────────────────────────────
    lsr_counts = sif_df["life_saving_rule"].value_counts()
    top_lsr    = lsr_counts.index[0] if len(lsr_counts) > 0 else "N/A"
    top_count  = lsr_counts.iloc[0] if len(lsr_counts) > 0 else 0

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🔴 Total SIF Reports", len(sif_df))
    with col2:
        st.metric("⚠️ Highest-Risk LSR", top_lsr)
    with col3:
        st.metric("📊 Top LSR Count", top_count)

    st.markdown("---")

    # ── Bar chart: SIF reports per LSR ───────────────────────────────────────
    lsr_df = lsr_counts.reset_index()
    lsr_df.columns = ["Life-Saving Rule", "SIF Reports"]
    lsr_colors = [LIFE_SAVING_RULES.get(r, {}).get("color", "#999") for r in lsr_df["Life-Saving Rule"]]
    lsr_icons  = [LIFE_SAVING_RULES.get(r, {}).get("icon", "📌") for r in lsr_df["Life-Saving Rule"]]
    lsr_df["Rule Label"] = [f"{icon} {rule}" for icon, rule in zip(lsr_icons, lsr_df["Life-Saving Rule"])]

    fig_lsr = px.bar(
        lsr_df,
        x="SIF Reports",
        y="Rule Label",
        orientation="h",
        color="SIF Reports",
        color_continuous_scale="Reds",
        text="SIF Reports",
        title="SIF-Potential Reports by IOGP Life-Saving Rule",
    )
    fig_lsr.update_traces(textposition="outside")
    fig_lsr.update_layout(
        height=480, showlegend=False, plot_bgcolor="#fafafa",
        xaxis_title="Number of SIF Reports",
        yaxis_title="",
        margin=dict(l=10, r=60, t=40, b=10),
    )
    st.plotly_chart(fig_lsr, use_container_width=True)

    # ── Treemap: LSR × Site ───────────────────────────────────────────────────
    st.markdown("##### LSR × Site Cross-Analysis")
    treemap_df = (
        sif_df.groupby(["life_saving_rule", "site"])["report_id"]
        .count()
        .reset_index()
        .rename(columns={"report_id": "count"})
    )
    treemap_df = treemap_df[treemap_df["count"] > 0]
    if not treemap_df.empty:
        fig_tree = px.treemap(
            treemap_df,
            path=["life_saving_rule", "site"],
            values="count",
            color="count",
            color_continuous_scale="Reds",
            title="SIF Reports: Life-Saving Rule → Site Breakdown",
        )
        fig_tree.update_layout(height=500)
        st.plotly_chart(fig_tree, use_container_width=True)

    # ── Trend: SIF reports by LSR over time ───────────────────────────────────
    st.markdown("##### 📈 SIF Trend by Life-Saving Rule (Monthly)")
    sif_df["date"] = pd.to_datetime(sif_df["date"], errors="coerce")
    sif_df["month"] = sif_df["date"].dt.to_period("M").astype(str)
    trend_df = (
        sif_df.groupby(["month", "life_saving_rule"])["report_id"]
        .count()
        .reset_index()
        .rename(columns={"report_id": "count"})
    )
    if not trend_df.empty and trend_df["month"].nunique() > 1:
        fig_trend = px.line(
            trend_df,
            x="month",
            y="count",
            color="life_saving_rule",
            markers=True,
            title="Monthly SIF Report Trend by Life-Saving Rule",
        )
        fig_trend.update_layout(height=400, xaxis_title="Month", yaxis_title="SIF Reports")
        st.plotly_chart(fig_trend, use_container_width=True)

    # ── LSR Rule Cards ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("##### 📖 IOGP Life-Saving Rules Reference")
    rule_cols = st.columns(3)
    for i, (rule_name, rule_data) in enumerate(LIFE_SAVING_RULES.items()):
        count = lsr_counts.get(rule_name, 0)
        with rule_cols[i % 3]:
            st.markdown(f"""
            <div style="background:white; border-radius:10px; padding:14px;
                        border-left:4px solid {rule_data['color']}; margin-bottom:10px;
                        box-shadow:0 1px 4px rgba(0,0,0,0.07)">
                <div style="font-size:1.5rem">{rule_data['icon']} <strong>{rule_name}</strong></div>
                <div style="color:#666; font-size:0.8rem; margin-top:4px">
                    {'; '.join(rule_data['keywords'][:4])}
                </div>
                <div style="margin-top:8px">
                    <span style="background:{rule_data['color']}; color:white; padding:2px 8px;
                                 border-radius:10px; font-size:0.8rem; font-weight:600">
                        {count} SIF Reports
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — PRECURSOR PATTERN EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════
def page_patterns():
    st.markdown('<div class="section-header">🔍 Precursor Pattern Explorer</div>', unsafe_allow_html=True)
    df = _get_results()
    if df is None:
        return

    patterns = st.session_state.get("patterns")

    if patterns is None or not patterns.get("topics"):
        st.info("No pattern data available. Please run analysis on the Upload & Analyze page first.")
        return

    sif_df = df[df["sif_potential"]].copy()

    # ── Pattern Summary Table ─────────────────────────────────────────────────
    st.markdown(f"**Method:** `{patterns.get('method', 'N/A')}` | "
                f"**SIF Reports Analysed:** `{len(sif_df)}`")

    from engine.pattern_miner import summarise_patterns
    pattern_df = summarise_patterns(patterns)

    if not pattern_df.empty:
        st.markdown("##### 🎯 Top Recurring Precursor Patterns")
        st.dataframe(pattern_df, use_container_width=True, hide_index=True)

        # ── Bar chart of pattern frequencies ─────────────────────────────────
        fig_patterns = px.bar(
            pattern_df.head(10),
            x="Report Count",
            y="Precursor Pattern",
            orientation="h",
            color="Report Count",
            color_continuous_scale="Reds",
            text="Report Count",
            title="Top 10 SIF Precursor Patterns (by Frequency)",
        )
        fig_patterns.update_traces(textposition="outside")
        fig_patterns.update_layout(
            height=450, showlegend=False,
            yaxis={"categoryorder": "total ascending"},
            margin=dict(l=10, r=60, t=40, b=10),
        )
        st.plotly_chart(fig_patterns, use_container_width=True)

    # ── Keyword word cloud simulation using bar chart ─────────────────────────
    st.markdown("---")
    st.markdown("##### ☁️ Precursor Keyword Frequency (SIF Reports)")
    from sklearn.feature_extraction.text import CountVectorizer
    if len(sif_df) >= 3:
        sif_texts = sif_df["narrative"].fillna("").tolist()
        vec = CountVectorizer(stop_words="english", ngram_range=(1, 2), max_features=30)
        try:
            X = vec.fit_transform(sif_texts)
            word_freq = pd.DataFrame({
                "keyword": vec.get_feature_names_out(),
                "count": X.toarray().sum(axis=0),
            }).sort_values("count", ascending=False)

            fig_wc = px.bar(
                word_freq.head(25),
                x="count", y="keyword",
                orientation="h",
                color="count",
                color_continuous_scale="OrRd",
                title="Most Frequent Terms in SIF-Potential Reports",
            )
            fig_wc.update_layout(
                height=500, showlegend=False,
                yaxis={"categoryorder": "total ascending"},
            )
            st.plotly_chart(fig_wc, use_container_width=True)
        except Exception:
            st.info("Keyword frequency chart unavailable.")

    # ── Activity / Location / Equipment Breakdown ─────────────────────────────
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    for col, entity_col, title in [
        (col1, "entities_activity",  "⚙️ Activities in SIF Reports"),
        (col2, "entities_location",  "📍 Locations in SIF Reports"),
        (col3, "entities_equipment", "🔧 Equipment in SIF Reports"),
    ]:
        if entity_col in sif_df.columns:
            all_vals = []
            for cell in sif_df[entity_col].fillna("").str.split(", "):
                all_vals.extend([v.strip() for v in cell if v.strip()])
            if all_vals:
                from collections import Counter
                counts = Counter(all_vals).most_common(10)
                mini_df = pd.DataFrame(counts, columns=["Entity", "Count"])
                with col:
                    st.markdown(f"**{title}**")
                    fig = px.bar(
                        mini_df, x="Count", y="Entity",
                        orientation="h", color="Count",
                        color_continuous_scale="Reds",
                    )
                    fig.update_layout(
                        height=350, showlegend=False,
                        yaxis={"categoryorder": "total ascending"},
                        margin=dict(l=5, r=30, t=10, b=5),
                    )
                    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — REPORT EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════
def page_explorer():
    st.markdown('<div class="section-header">📋 Report Explorer</div>', unsafe_allow_html=True)
    df = _get_results()
    if df is None:
        return

    # ── Filters ───────────────────────────────────────────────────────────────
    with st.expander("🔍 Search & Filter", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            sif_filter = st.selectbox("SIF Status", ["All", "SIF-Potential Only", "Non-SIF Only"])
        with col2:
            conf_filter = st.multiselect("Confidence", ["HIGH", "MEDIUM", "LOW"], default=[])
        with col3:
            lsr_filter = st.multiselect(
                "Life-Saving Rule",
                sorted(df["life_saving_rule"].unique().tolist()),
                default=[],
            )
        with col4:
            search_text = st.text_input("🔎 Keyword search in narrative")

    filtered = df.copy()
    if sif_filter == "SIF-Potential Only":
        filtered = filtered[filtered["sif_potential"]]
    elif sif_filter == "Non-SIF Only":
        filtered = filtered[~filtered["sif_potential"]]
    if conf_filter:
        filtered = filtered[filtered["confidence"].isin(conf_filter)]
    if lsr_filter:
        filtered = filtered[filtered["life_saving_rule"].isin(lsr_filter)]
    if search_text:
        filtered = filtered[
            filtered["narrative"].str.contains(search_text, case=False, na=False)
        ]

    st.caption(f"Showing {len(filtered)} of {len(df)} reports")

    # ── Sortable Table ────────────────────────────────────────────────────────
    display_cols = [
        "report_id", "date", "site", "report_type",
        "sif_potential", "sif_score", "confidence",
        "life_saving_rule", "top_signals",
    ]
    display_df = filtered[display_cols].copy()
    display_df["sif_score"] = display_df["sif_score"].round(3)
    display_df = display_df.rename(columns={
        "report_id": "Report ID", "date": "Date", "site": "Site",
        "report_type": "Type", "sif_potential": "SIF?",
        "sif_score": "Score", "confidence": "Confidence",
        "life_saving_rule": "Life-Saving Rule", "top_signals": "Signals",
    })

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Score": st.column_config.ProgressColumn(
                "SIF Score", min_value=0, max_value=1, format="%.3f"
            ),
            "SIF?": st.column_config.CheckboxColumn("SIF?"),
        },
    )

    # ── Detail View ───────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("##### 🔬 Drill-Down: Individual Report")
    if not filtered.empty:
        selected_id = st.selectbox(
            "Select Report ID",
            filtered["report_id"].tolist(),
        )
        row = filtered[filtered["report_id"] == selected_id].iloc[0]

        sif_color = "#e74c3c" if row["sif_potential"] else "#27ae60"
        sif_label = "✅ SIF-POTENTIAL" if row["sif_potential"] else "✅ NON-SIF"

        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"""
            <div style="background:white; border-radius:12px; padding:20px; box-shadow:0 2px 8px rgba(0,0,0,0.08);">
                <div style="font-size:1.1rem; font-weight:600; color:#1a3a5c; margin-bottom:10px;">
                    📄 {row['report_id']} | {row['site']} | {row['date']}
                </div>
                <div style="background:#f8f9fa; border-radius:8px; padding:14px; 
                            font-size:0.95rem; line-height:1.7; color:#333;">
                    {row['narrative']}
                </div>
                <div style="margin-top:12px; font-style:italic; color:#555; font-size:0.85rem;">
                    💬 {row['explanation']}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div style="background:white; border-radius:12px; padding:20px;
                        box-shadow:0 2px 8px rgba(0,0,0,0.08); text-align:center;">
                <div style="font-size:1rem; font-weight:700; color:{sif_color}; 
                            background:{sif_color}22; padding:10px; border-radius:8px; margin-bottom:12px;">
                    {sif_label}
                </div>
                <div style="font-size:2rem; font-weight:800; color:{sif_color}">
                    {row['sif_score']:.3f}
                </div>
                <div style="color:#999; font-size:0.75rem">SIF Score</div>
                <hr/>
                <div style="font-weight:600">{row['confidence']}</div>
                <div style="color:#999; font-size:0.75rem">Confidence</div>
                <hr/>
                <div style="font-weight:600; color:#1a3a5c">{row['life_saving_rule']}</div>
                <div style="color:#999; font-size:0.75rem">Life-Saving Rule</div>
                <hr/>
                <div style="font-size:0.8rem; color:#555">
                    🔑 <em>{row['top_signals'] or 'No key signals'}</em>
                </div>
            </div>
            """, unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _get_results() -> pd.DataFrame | None:
    df = st.session_state.get("results_df")
    if df is None or df.empty:
        st.warning("⚠️ No data loaded yet. Please go to **📤 Upload & Analyze** to load and process reports.")
        if st.button("Go to Upload & Analyze"):
            st.session_state["_nav"] = "📤 Upload & Analyze"
            st.rerun()
        return None
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# ROUTER
# ═══════════════════════════════════════════════════════════════════════════════
if page == "📤 Upload & Analyze":
    page_upload()
elif page == "🔴 SIF Risk Heatmap":
    page_heatmap()
elif page == "🏷️ Life-Saving Rules":
    page_lsr()
elif page == "🔍 Pattern Explorer":
    page_patterns()
elif page == "📋 Report Explorer":
    page_explorer()
