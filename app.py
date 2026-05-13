import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))
from process_engine import (
    detect_churn_column, preprocess, train_models,
    predict_batch, classify_risk, get_feature_importance, generate_recommendations
)

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnLabs · AI Churn Intelligence",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Dark background */
.stApp { background: #0a0f1e; color: #e2e8f0; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1424 0%, #111827 100%);
    border-right: 1px solid #1e293b;
}
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }

/* Hide default header */
header[data-testid="stHeader"] { background: transparent; }

/* Metric cards */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1rem 1.2rem;
}
[data-testid="stMetricLabel"] { color: #94a3b8 !important; font-size: 0.78rem !important; }
[data-testid="stMetricValue"] { color: #f1f5f9 !important; font-size: 1.6rem !important; font-weight: 700 !important; }

/* Tabs */
[data-testid="stTabs"] button {
    color: #94a3b8 !important;
    font-weight: 500;
    border-radius: 8px 8px 0 0;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #a78bfa !important;
    border-bottom: 2px solid #7c3aed !important;
    background: #1e293b !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #6d28d9);
    color: white !important;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    padding: 0.55rem 2rem;
    transition: all 0.2s ease;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #8b5cf6, #7c3aed);
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(124, 58, 237, 0.4);
}

/* Download button */
.stDownloadButton > button {
    background: linear-gradient(135deg, #0f766e, #0d9488) !important;
    color: white !important;
    border: none;
    border-radius: 10px;
    font-weight: 600;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: #1e293b;
    border: 2px dashed #334155;
    border-radius: 12px;
    padding: 1rem;
}

/* Dataframe */
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

/* Info / warning boxes */
.stAlert { border-radius: 10px; }

/* Selectbox */
[data-testid="stSelectbox"] > div > div {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 8px;
    color: #e2e8f0;
}

/* Risk badge helper classes */
.risk-high   { color: #f87171; font-weight: 700; }
.risk-medium { color: #fbbf24; font-weight: 700; }
.risk-low    { color: #34d399; font-weight: 700; }

/* Hero header */
.hero-title {
    font-size: 2.4rem; font-weight: 800;
    background: linear-gradient(135deg, #a78bfa, #818cf8, #38bdf8);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    line-height: 1.2; margin-bottom: 0.3rem;
}
.hero-sub { color: #64748b; font-size: 1rem; margin-bottom: 1.5rem; }

/* Card container */
.card {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid #334155; border-radius: 14px;
    padding: 1.4rem 1.6rem; margin-bottom: 1rem;
}

/* Section header */
.section-hdr {
    font-size: 1.1rem; font-weight: 700;
    color: #a78bfa; margin: 1.2rem 0 0.6rem 0;
    border-left: 3px solid #7c3aed; padding-left: 0.7rem;
}

div[data-testid="stVerticalBlock"] > div:has(div.stProgress) {}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 0.5rem 0;'>
        <div style='font-size:2.2rem'>🔬</div>
        <div style='font-size:1.3rem; font-weight:800; color:#a78bfa;'>ChurnLabs</div>
        <div style='font-size:0.75rem; color:#475569;'>AI Churn Intelligence Platform</div>
    </div>
    <hr style='border-color:#1e293b; margin: 0.8rem 0;'>
    """, unsafe_allow_html=True)

    st.markdown("**📋 How it works**")
    st.markdown("""
    <div style='font-size:0.82rem; color:#64748b; line-height:1.8;'>
    1️⃣ Upload your customer CSV<br>
    2️⃣ Map the churn column<br>
    3️⃣ Train ML models<br>
    4️⃣ Get predictions & advice
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#1e293b;'>", unsafe_allow_html=True)
    st.markdown("**📥 Sample Dataset**")
    sample_path = os.path.join(os.path.dirname(__file__), "data", "sample_dataset.csv")
    if os.path.exists(sample_path):
        with open(sample_path, "rb") as f:
            st.download_button(
                "⬇️ Download Sample CSV",
                data=f,
                file_name="sample_churn_dataset.csv",
                mime="text/csv",
                use_container_width=True,
            )

    st.markdown("<hr style='border-color:#1e293b;'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.72rem; color:#334155; text-align:center;'>
    Models: Logistic Regression · Random Forest · XGBoost<br><br>
    © 2025 ChurnLabs
    </div>
    """, unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='hero-title'>ChurnLabs · AI Churn Intelligence</div>
<div class='hero-sub'>Upload your dataset → Train models → Predict churn → Retain customers</div>
""", unsafe_allow_html=True)

# ── Session State ─────────────────────────────────────────────────────────────
for key in ["df_raw", "df_processed", "y", "feature_names", "trained_models",
            "scaler", "model_results", "best_model_name", "predictions_df",
            "churn_col", "id_col", "encoders"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📂  Upload & Train",
    "🔮  Predictions",
    "📊  EDA & Insights",
    "💡  Retention Advisor",
])

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — UPLOAD & TRAIN
# ════════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("<div class='section-hdr'>Upload Your Dataset</div>", unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Drop your customer CSV here — any churn dataset works!",
        type=["csv"],
        label_visibility="collapsed",
    )

    if uploaded:
        df_raw = pd.read_csv(uploaded)
        st.session_state["df_raw"] = df_raw

        st.success(f"✅ Loaded **{len(df_raw):,} rows** × **{len(df_raw.columns)} columns**")

        with st.expander("👁️ Preview Data (first 8 rows)", expanded=True):
            st.dataframe(df_raw.head(8), use_container_width=True)

        # ── Auto-detect churn column & ID column (no dropdowns) ──────────────
        cols = list(df_raw.columns)

        def _detect_churn_col(columns):
            for c in columns:
                if c.lower() == "churn": return c
            for c in columns:
                if "churn" in c.lower(): return c
            for kw in ["attrition", "churned", "left", "target", "label"]:
                for c in columns:
                    if kw in c.lower(): return c
            return columns[-1]  # last column fallback

        def _detect_id_col(columns):
            for c in columns:
                if any(k in c.lower() for k in ["id", "customer", "uid"]):
                    return c
            return None

        churn_col = _detect_churn_col(cols)
        id_col    = _detect_id_col(cols)

        # Show detected columns as info badges
        ci1, ci2 = st.columns(2)
        ci1.info(f"🎯 **Churn label column:** `{churn_col}`")
        if id_col:
            ci2.info(f"🪪 **Customer ID column:** `{id_col}`")
        else:
            ci2.info("🪪 **Customer ID:** not detected — using row index")

        st.markdown("<div class='section-hdr'>Train Models</div>", unsafe_allow_html=True)

        if st.button("🚀 Train All Models", use_container_width=False):
            with st.spinner("Training Logistic Regression, Random Forest & XGBoost..."):
                try:
                    X, y, feature_names, encoders = preprocess(df_raw, churn_col, id_col)
                    trained, scaler, results, best_name = train_models(X, y)

                    st.session_state.update({
                        "df_processed": X,
                        "y": y,
                        "feature_names": feature_names,
                        "encoders": encoders,
                        "trained_models": trained,
                        "scaler": scaler,
                        "model_results": results,
                        "best_model_name": best_name,
                        "churn_col": churn_col,
                        "id_col": id_col,
                    })
                    st.success(f"✅ Training complete! Best model: **{best_name}**")
                except Exception as e:
                    st.error(f"❌ Training failed: {e}")

    if st.session_state["model_results"]:
        st.markdown("<div class='section-hdr'>Model Comparison</div>", unsafe_allow_html=True)
        results = st.session_state["model_results"]
        best    = st.session_state["best_model_name"]

        metrics_rows = []
        for mname, mdict in results.items():
            row = {"Model": ("⭐ " if mname == best else "") + mname}
            row.update({k: f"{v:.4f}" for k, v in mdict.items() if k != "Confusion Matrix"})
            metrics_rows.append(row)

        metrics_df = pd.DataFrame(metrics_rows).set_index("Model")
        st.dataframe(metrics_df, use_container_width=True)

        # Metric cards for best model
        bm = results[best]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Accuracy",  f"{bm['Accuracy']:.2%}")
        c2.metric("Precision", f"{bm['Precision']:.2%}")
        c3.metric("Recall",    f"{bm['Recall']:.2%}")
        c4.metric("F1 Score",  f"{bm['F1 Score']:.2%}")
        c5.metric("ROC-AUC",   f"{bm['ROC-AUC']:.2%}")

# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — PREDICTIONS
# ════════════════════════════════════════════════════════════════════════════════
with tab2:
    if not st.session_state["trained_models"]:
        st.info("👈 Please upload a dataset and train models first (Tab 1).")
    else:
        st.markdown("<div class='section-hdr'>Batch Churn Predictions</div>", unsafe_allow_html=True)

        best_name  = st.session_state["best_model_name"]
        best_model = st.session_state["trained_models"][best_name]
        scaler     = st.session_state["scaler"]
        X          = st.session_state["df_processed"]
        df_raw     = st.session_state["df_raw"]
        id_col     = st.session_state["id_col"]
        churn_col  = st.session_state["churn_col"]

        use_scaled = best_name == "Logistic Regression"
        proba = predict_batch(best_model, scaler, X, use_scaled=use_scaled)

        pred_df = pd.DataFrame()
        if id_col and id_col in df_raw.columns:
            pred_df["Customer ID"] = df_raw[id_col].values
        else:
            pred_df["Customer ID"] = [f"#{i+1}" for i in range(len(df_raw))]

        pred_df["Churn Probability"] = (proba * 100).round(1)
        pred_df["Risk Level"] = [classify_risk(p) for p in proba]
        pred_df["Predicted Churn"] = ["Yes" if p >= 0.5 else "No" for p in proba]

        if churn_col and churn_col in df_raw.columns:
            pred_df["Actual Churn"] = df_raw[churn_col].values

        st.session_state["predictions_df"] = pred_df

        # Summary KPIs
        total     = len(pred_df)
        high_risk = (proba >= 0.70).sum()
        med_risk  = ((proba >= 0.40) & (proba < 0.70)).sum()
        low_risk  = (proba < 0.40).sum()
        avg_prob  = proba.mean()

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Customers", f"{total:,}")
        c2.metric("🔴 High Risk",    f"{high_risk:,}", f"{high_risk/total:.1%}")
        c3.metric("🟡 Medium Risk",  f"{med_risk:,}",  f"{med_risk/total:.1%}")
        c4.metric("🟢 Low Risk",     f"{low_risk:,}",  f"{low_risk/total:.1%}")
        c5.metric("Avg Churn Prob",  f"{avg_prob:.1%}")

        st.markdown("<div class='section-hdr'>Results Table</div>", unsafe_allow_html=True)

        risk_filter = st.selectbox(
            "Filter by Risk Level",
            ["All", "🔴 High", "🟡 Medium", "🟢 Low"],
            label_visibility="collapsed",
        )
        display_df = pred_df if risk_filter == "All" else pred_df[pred_df["Risk Level"] == risk_filter]
        st.dataframe(
            display_df.sort_values("Churn Probability", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "⬇️ Export Predictions CSV",
            data=pred_df.to_csv(index=False).encode(),
            file_name="churn_predictions.csv",
            mime="text/csv",
        )

# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — EDA & INSIGHTS
# ════════════════════════════════════════════════════════════════════════════════
with tab3:
    if st.session_state["df_raw"] is None:
        st.info("👈 Please upload a dataset first (Tab 1).")
    else:
        df_raw    = st.session_state["df_raw"]
        churn_col = st.session_state["churn_col"]

        PLOTLY_LAYOUT = dict(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.6)",
            font_color="#94a3b8",
            title_font_color="#e2e8f0",
            colorway=["#7c3aed","#06b6d4","#10b981","#f59e0b","#ef4444"],
        )

        st.markdown("<div class='section-hdr'>Data Overview</div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Rows",    f"{len(df_raw):,}")
        c2.metric("Total Columns", f"{len(df_raw.columns)}")
        if churn_col and churn_col in df_raw.columns:
            churn_vals = df_raw[churn_col].astype(str).str.lower().str.strip()
            churn_rate = churn_vals.isin(["yes","1","true","churn"]).mean()
            c3.metric("Churn Rate", f"{churn_rate:.1%}")

        if churn_col and churn_col in df_raw.columns:
            row1_l, row1_r = st.columns(2)

            # Churn distribution pie
            with row1_l:
                churn_counts = df_raw[churn_col].value_counts().reset_index()
                churn_counts.columns = ["Churn", "Count"]
                fig = px.pie(
                    churn_counts, names="Churn", values="Count",
                    title="Churn Distribution",
                    color_discrete_sequence=["#7c3aed","#06b6d4"],
                    hole=0.55,
                )
                fig.update_layout(**PLOTLY_LAYOUT)
                st.plotly_chart(fig, use_container_width=True)

            # Churn by Contract (if exists)
            with row1_r:
                contract_col = next((c for c in df_raw.columns if "contract" in c.lower()), None)
                if contract_col:
                    grp = df_raw.groupby(contract_col)[churn_col].apply(
                        lambda x: (x.astype(str).str.lower().str.strip().isin(["yes","1","true","churn"])).mean()
                    ).reset_index()
                    grp.columns = [contract_col, "Churn Rate"]
                    fig2 = px.bar(
                        grp, x=contract_col, y="Churn Rate",
                        title="Churn Rate by Contract Type",
                        color="Churn Rate",
                        color_continuous_scale=["#1e293b","#7c3aed","#ef4444"],
                        text_auto=".1%",
                    )
                    fig2.update_layout(**PLOTLY_LAYOUT)
                    st.plotly_chart(fig2, use_container_width=True)

            row2_l, row2_r = st.columns(2)

            # Tenure histogram
            with row2_l:
                tenure_col = next((c for c in df_raw.columns if "tenure" in c.lower()), None)
                if tenure_col:
                    fig3 = px.histogram(
                        df_raw, x=tenure_col, color=churn_col,
                        title="Tenure Distribution by Churn",
                        barmode="overlay", opacity=0.75,
                        color_discrete_sequence=["#06b6d4","#ef4444"],
                    )
                    fig3.update_layout(**PLOTLY_LAYOUT)
                    st.plotly_chart(fig3, use_container_width=True)

            # Monthly Charges box plot
            with row2_r:
                mc_col = next((c for c in df_raw.columns if "monthly" in c.lower() and "charge" in c.lower()), None)
                if mc_col:
                    fig4 = px.box(
                        df_raw, x=churn_col, y=mc_col,
                        title="Monthly Charges vs Churn",
                        color=churn_col,
                        color_discrete_sequence=["#10b981","#ef4444"],
                    )
                    fig4.update_layout(**PLOTLY_LAYOUT)
                    st.plotly_chart(fig4, use_container_width=True)

        # Feature Importance
        if st.session_state["trained_models"] and st.session_state["best_model_name"]:
            st.markdown("<div class='section-hdr'>Feature Importance</div>", unsafe_allow_html=True)
            best  = st.session_state["best_model_name"]
            model = st.session_state["trained_models"][best]
            feat_df = get_feature_importance(model, st.session_state["feature_names"])
            if not feat_df.empty:
                top15 = feat_df.head(15)
                fig5 = px.bar(
                    top15, x="Importance", y="Feature",
                    orientation="h",
                    title=f"Top Features ({best})",
                    color="Importance",
                    color_continuous_scale=["#1e293b","#7c3aed","#a78bfa"],
                )
                fig5.update_layout(**PLOTLY_LAYOUT, yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(fig5, use_container_width=True)

            # Confusion Matrix
            cm = st.session_state["model_results"][best]["Confusion Matrix"]
            cm_arr = np.array(cm)
            fig6 = px.imshow(
                cm_arr,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=["Not Churn", "Churn"], y=["Not Churn", "Churn"],
                color_continuous_scale="Purples",
                title=f"Confusion Matrix — {best}",
                text_auto=True,
            )
            fig6.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig6, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 4 — RETENTION ADVISOR
# ════════════════════════════════════════════════════════════════════════════════
with tab4:
    if st.session_state["predictions_df"] is None:
        st.info("👈 Please run predictions first (Tab 2).")
    else:
        pred_df = st.session_state["predictions_df"]
        df_raw  = st.session_state["df_raw"]

        st.markdown("<div class='section-hdr'>Retention Action Plan</div>", unsafe_allow_html=True)

        high_risk_ids = pred_df[pred_df["Churn Probability"] >= 70]["Customer ID"].values
        st.markdown(f"Showing retention advice for **{len(high_risk_ids)} high-risk customers** (churn prob ≥ 70%)")

        if len(high_risk_ids) == 0:
            st.success("🎉 No high-risk customers found! Your customer base looks healthy.")
        else:
            id_col    = st.session_state["id_col"]
            action_rows = []

            for cid in high_risk_ids[:50]:  # cap at 50 for performance
                # Find original row
                if id_col and id_col in df_raw.columns:
                    mask = df_raw[id_col].astype(str) == str(cid)
                    row  = df_raw[mask].iloc[0] if mask.any() else pd.Series()
                else:
                    try:
                        idx = int(str(cid).replace("#", "")) - 1
                        row = df_raw.iloc[idx]
                    except Exception:
                        row = pd.Series()

                recs = generate_recommendations(row)
                prob = float(pred_df[pred_df["Customer ID"].astype(str) == str(cid)]["Churn Probability"].iloc[0])
                action_rows.append({
                    "Customer ID": cid,
                    "Churn Prob": f"{prob:.1f}%",
                    "Top Recommendation": recs[0],
                    "Additional Actions": len(recs) - 1,
                })

            action_df = pd.DataFrame(action_rows)
            st.dataframe(action_df, use_container_width=True, hide_index=True)

            # Expandable per-customer details
            st.markdown("<div class='section-hdr'>Detailed Recommendations</div>", unsafe_allow_html=True)
            for _, arow in action_df.iterrows():
                cid  = arow["Customer ID"]
                prob = arow["Churn Prob"]

                if id_col and id_col in df_raw.columns:
                    mask = df_raw[id_col].astype(str) == str(cid)
                    row  = df_raw[mask].iloc[0] if mask.any() else pd.Series()
                else:
                    try:
                        idx = int(str(cid).replace("#", "")) - 1
                        row = df_raw.iloc[idx]
                    except Exception:
                        row = pd.Series()

                recs = generate_recommendations(row)
                with st.expander(f"🔴 {cid}  —  Churn Probability: {prob}"):
                    for i, rec in enumerate(recs, 1):
                        st.markdown(f"**{i}.** {rec}")

            # Export action plan
            st.download_button(
                "⬇️ Export Action Plan CSV",
                data=action_df.to_csv(index=False).encode(),
                file_name="retention_action_plan.csv",
                mime="text/csv",
            )

        # Segment-level strategy
        st.markdown("<div class='section-hdr'>Segment Strategy Guide</div>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div class='card'>
            <b style='color:#ef4444;'>🔴 High Risk (≥70%)</b><br><br>
            • Immediate outreach call<br>
            • Offer personalised discount<br>
            • Assign dedicated account manager<br>
            • Propose contract upgrade incentive
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class='card'>
            <b style='color:#fbbf24;'>🟡 Medium Risk (40–70%)</b><br><br>
            • Send satisfaction survey<br>
            • Promote loyalty rewards<br>
            • Offer free service add-on trial<br>
            • Targeted email campaign
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div class='card'>
            <b style='color:#34d399;'>🟢 Low Risk (&lt;40%)</b><br><br>
            • Cross-sell premium features<br>
            • Maintain current service SLA<br>
            • Encourage referral program<br>
            • Upsell complementary services
            </div>
            """, unsafe_allow_html=True)
