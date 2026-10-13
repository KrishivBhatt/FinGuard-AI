"""
FinGuard AI - Main Entry Point & Executive Dashboard (Page 1)
Third-Year Computer Science Diploma Semester Mini-Project
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import config
from src.ui_components import apply_fintech_theme, render_sidebar_status, render_app_footer
from src.data_loader import (
    load_dataset_from_path,
    validate_fraud_dataset,
    get_dataset_summary,
    generate_synthetic_demo_data
)
from src.train import load_training_metadata
from src.visualisations import create_amount_distribution_chart

# Page configuration
st.set_page_config(
    page_title="FinGuard AI — Intelligent Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply fintech styling
apply_fintech_theme()

# Sidebar persistent status and model selection
selected_model = render_sidebar_status()

# Session State for Dataset
if "current_dataset" not in st.session_state:
    if config.REAL_DATASET_PATH.exists():
        df_real, _ = load_dataset_from_path(config.REAL_DATASET_PATH)
        st.session_state["current_dataset"] = df_real
        st.session_state["dataset_source"] = "Real Kaggle Dataset (data/creditcard.csv)"
    elif config.SAMPLE_DATASET_PATH.exists():
        df_sample, _ = load_dataset_from_path(config.SAMPLE_DATASET_PATH)
        st.session_state["current_dataset"] = df_sample
        st.session_state["dataset_source"] = "Synthetic Demo Dataset (data/sample_transactions.csv)"
    else:
        st.session_state["current_dataset"] = None
        st.session_state["dataset_source"] = "None"

# Header Banner
st.markdown("""
<div class="fintech-header">
    <h1 class="fintech-title">🛡️ FinGuard AI</h1>
    <div class="fintech-subtitle">
        Intelligent Credit Card Fraud Detection & Personal Spending Analytics System
    </div>
</div>
""", unsafe_allow_html=True)

# Project Introduction & Purpose
col_intro, col_status = st.columns([2, 1])

with col_intro:
    st.markdown("""
    ### Executive Overview
    **FinGuard AI** is a machine learning-driven financial security platform engineered to detect fraudulent credit card transactions in real-time while providing intelligent personal spending analytics.
    
    - **Genuine ML Pipeline:** Compares **Logistic Regression**, **Decision Trees**, and **Random Forest Classifiers** with zero data leakage.
    - **Imbalance-Aware:** Built specifically for extreme class imbalances (fraud < 0.2%) using Stratified Splitting, PR-AUC, and balanced class weights.
    - **Explainable Decisions:** Inspect transaction feature anomalies and tune classification thresholds dynamically.
    """)

with col_status:
    st.markdown("### System Pipeline Status")
    metadata = load_training_metadata()
    if metadata:
        rec_model = metadata.get("recommended_model", "N/A").replace("_", " ").title()
        is_synth = metadata.get("is_synthetic", False)
        badge = '<span class="status-badge badge-warning">Synthetic Demo</span>' if is_synth else '<span class="status-badge badge-success">Production Data</span>'
        st.markdown(f"**Trained Models:** Active {badge}", unsafe_allow_html=True)
        st.markdown(f"**Optimal Model:** `{rec_model}`")
        st.markdown(f"**Evaluated Test Records:** `{metadata.get('test_rows', 0):,}`")
        st.markdown(f"**Last Trained:** `{metadata.get('timestamp', 'N/A')[:19]}`")
    else:
        st.markdown('<span class="status-badge badge-danger">Models Not Trained</span>', unsafe_allow_html=True)
        st.warning("No trained models found. Visit the **Model Lab** to train pipelines.")

st.markdown("---")

# Dataset Management Section
st.subheader("📊 Dataset Ingestion & Status Panel")

ds_col1, ds_col2, ds_col3 = st.columns([1.2, 1.2, 1.6])

with ds_col1:
    st.markdown("##### Upload New CSV")
    uploaded_file = st.file_uploader("Upload credit card dataset (CSV):", type=["csv"], key="dash_uploader")
    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)
            is_valid, errors, warnings = validate_fraud_dataset(uploaded_df, require_target=True)
            if is_valid:
                st.session_state["current_dataset"] = uploaded_df
                st.session_state["dataset_source"] = f"Uploaded File: {uploaded_file.name}"
                st.success(f"Loaded {len(uploaded_df):,} records successfully!")
            else:
                st.error("Validation failed:\n" + "\n".join(errors))
        except Exception as e:
            st.error(f"Failed to parse CSV: {e}")

with ds_col2:
    st.markdown("##### Quick Load Options")
    if config.REAL_DATASET_PATH.exists():
        if st.button("Load Local Kaggle Dataset", use_container_width=True):
            df_k, res = load_dataset_from_path(config.REAL_DATASET_PATH)
            st.session_state["current_dataset"] = df_k
            st.session_state["dataset_source"] = "Real Kaggle Dataset (data/creditcard.csv)"
            st.rerun()

    if st.button("Generate Synthetic Demo Dataset (5,000 rows)", use_container_width=True):
        synth_df = generate_synthetic_demo_data(n_samples=5000, fraud_ratio=0.018)
        st.session_state["current_dataset"] = synth_df
        st.session_state["dataset_source"] = "Synthetically Generated Demo Data"
        st.rerun()

with ds_col3:
    st.markdown("##### Active Dataset Information")
    st.info(f"**Current Source:** {st.session_state.get('dataset_source', 'None')}")
    if "Synthetic" in st.session_state.get("dataset_source", ""):
        st.caption("⚠️ **Notice:** Synthetic data is provided for UI demonstration and development testing only. Real benchmark metrics require `creditcard.csv`.")

# Real Dataset Metrics
current_df = st.session_state.get("current_dataset")
summary = get_dataset_summary(current_df)

st.markdown("### 📈 Live Transaction Telemetry")
m1, m2, m3, m4, m5 = st.columns(5)

with m1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Total Transactions</div>
        <div class="metric-value">{summary['total_records']:,}</div>
        <div class="metric-sub">Analyzed in memory</div>
    </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Legitimate Count</div>
        <div class="metric-value" style="color: #10B981;">{summary['legit_count']:,}</div>
        <div class="metric-sub">Class 0 verified</div>
    </div>
    """, unsafe_allow_html=True)

with m3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Fraudulent Count</div>
        <div class="metric-value" style="color: #EF4444;">{summary['fraud_count']:,}</div>
        <div class="metric-sub">Class 1 flagged</div>
    </div>
    """, unsafe_allow_html=True)

with m4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Fraud Prevalence</div>
        <div class="metric-value" style="color: #F59E0B;">{summary['fraud_percentage']}%</div>
        <div class="metric-sub">Ratio {summary['imbalance_ratio']}</div>
    </div>
    """, unsafe_allow_html=True)

with m5:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Active Model</div>
        <div class="metric-value" style="font-size: 1.25rem; color: #3B82F6;">{selected_model.replace('_', ' ').title()}</div>
        <div class="metric-sub">Configured in sidebar</div>
    </div>
    """, unsafe_allow_html=True)

# Visual Distribution & Navigation
if current_df is not None and not current_df.empty:
    st.markdown("### 📊 Distribution of Transaction Values")
    fig = create_amount_distribution_chart(current_df)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("No dataset loaded yet. Upload or generate a demo dataset to see telemetry charts.")

# Quick Navigation Hub
st.markdown("---")
st.markdown("### 🧭 Quick Navigation Hub")

nav1, nav2, nav3, nav4 = st.columns(4)

with nav1:
    st.markdown("""
    #### 🔍 [1. Fraud Detection](Fraud_Detection)
    Test live transactions, adjust the detection threshold, and inspect probability scores.
    """)

with nav2:
    st.markdown("""
    #### 📋 [2. Transaction Explorer](Transaction_Explorer)
    Filter transactions by amount, actual class, or model prediction with CSV export.
    """)

with nav3:
    st.markdown("""
    #### 🔬 [3. Model Lab](Model_Lab)
    Compare Logistic Regression, Decision Tree, and Random Forest models side-by-side.
    """)

with nav4:
    st.markdown("""
    #### 💳 [4. Spending Analytics](Spending_Analytics)
    Analyze personal expenses, monthly trends, category budgets, and unusual transactions.
    """)

render_app_footer()
