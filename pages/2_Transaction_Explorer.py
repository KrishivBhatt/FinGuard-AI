"""
FinGuard AI - Transaction Explorer (Page 3)
Searchable, filterable transaction analytics table with pagination and CSV export.
"""

import streamlit as st
import pandas as pd
import numpy as np
import config
from src.ui_components import apply_fintech_theme, render_sidebar_status, render_app_footer
from src.predict import predict_batch_transactions
from src.data_loader import generate_synthetic_demo_data

st.set_page_config(
    page_title="Transaction Explorer — FinGuard AI",
    page_icon="🛡️",
    layout="wide"
)

apply_fintech_theme()
selected_model = render_sidebar_status()

# Header
st.markdown("""
<div class="fintech-header">
    <h1 class="fintech-title">📋 Transaction Explorer</h1>
    <div class="fintech-subtitle">
        Filter, search, inspect, and export live records with side-by-side ground truth vs ML predictions.
    </div>
</div>
""", unsafe_allow_html=True)

# Load dataset from session state or generate
current_df = st.session_state.get("current_dataset")
if current_df is None or current_df.empty:
    st.info("No active dataset in memory. Loading sample records for exploration...")
    current_df = generate_synthetic_demo_data(n_samples=500, fraud_ratio=0.03)
    st.session_state["current_dataset"] = current_df

# Ensure predictions exist for exploration
if "Fraud_Probability" not in current_df.columns:
    with st.spinner("Scoring dataset records with active model..."):
        try:
            current_df = predict_batch_transactions(current_df, model_key=selected_model)
            st.session_state["current_dataset"] = current_df
        except Exception as e:
            st.warning(f"Could not compute batch predictions: {e}")

# Filter Controls
st.markdown("### 🔍 Filters & Search Parameters")
f_col1, f_col2, f_col3, f_col4 = st.columns(4)

with f_col1:
    actual_filter = st.selectbox(
        "Ground Truth Class:",
        ["All Records", "Legitimate Only (0)", "Fraudulent Only (1)"] if "Class" in current_df.columns else ["N/A"]
    )

with f_col2:
    pred_filter = st.selectbox(
        "Model Prediction:",
        ["All Predictions", "Predicted Legitimate (0)", "Predicted Fraud (1)"] if "Predicted_Class" in current_df.columns else ["N/A"]
    )

with f_col3:
    max_data_amt = float(current_df["Amount"].max()) if "Amount" in current_df.columns else 1000.0
    amt_range = st.slider(
        "Transaction Amount Range:",
        min_value=0.0,
        max_value=max(100.0, max_data_amt),
        value=(0.0, max_data_amt),
        step=5.0
    )

with f_col4:
    prob_min = st.slider(
        "Min Fraud Probability (%):",
        min_value=0,
        max_value=100,
        value=0,
        step=5
    )

# Apply filters
filtered_df = current_df.copy()

if "Amount" in filtered_df.columns:
    filtered_df = filtered_df[
        (filtered_df["Amount"] >= amt_range[0]) & (filtered_df["Amount"] <= amt_range[1])
    ]

if "Class" in filtered_df.columns:
    if actual_filter == "Legitimate Only (0)":
        filtered_df = filtered_df[filtered_df["Class"] == 0]
    elif actual_filter == "Fraudulent Only (1)":
        filtered_df = filtered_df[filtered_df["Class"] == 1]

if "Predicted_Class" in filtered_df.columns:
    if pred_filter == "Predicted Legitimate (0)":
        filtered_df = filtered_df[filtered_df["Predicted_Class"] == 0]
    elif pred_filter == "Predicted Fraud (1)":
        filtered_df = filtered_df[filtered_df["Predicted_Class"] == 1]

if "Fraud_Probability" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["Fraud_Probability"] >= (prob_min / 100.0)]

# Summary stats on filtered records
st.markdown("---")
s1, s2, s3, s4 = st.columns(4)
s1.metric("Matching Records", f"{len(filtered_df):,}")
s2.metric("Total Amount", f"${filtered_df['Amount'].sum():,.2f}" if "Amount" in filtered_df.columns else "N/A")
s3.metric("Average Amount", f"${filtered_df['Amount'].mean():,.2f}" if "Amount" in filtered_df.columns and len(filtered_df) > 0 else "N/A")

if "Class" in filtered_df.columns and "Predicted_Class" in filtered_df.columns and len(filtered_df) > 0:
    agreement = (filtered_df["Class"] == filtered_df["Predicted_Class"]).mean() * 100.0
    s4.metric("Model Agreement on Filtered", f"{agreement:.1f}%")
else:
    s4.metric("Model Agreement", "N/A")

# Pagination
st.markdown("### 📄 Filtered Data Records")
page_size = st.selectbox("Rows per page:", [25, 50, 100, 200], index=1)
total_pages = max(1, int(np.ceil(len(filtered_df) / page_size)))
current_page = st.number_input("Page:", min_value=1, max_value=total_pages, value=1, step=1)

start_idx = (current_page - 1) * page_size
end_idx = start_idx + page_size
paged_df = filtered_df.iloc[start_idx:end_idx]

# Display column priority: Time, Amount, Class (Ground Truth), Predicted_Class, Fraud_Probability, then PCA features
display_cols = []
for c in ["Time", "Amount", "Class", "Predicted_Class", "Fraud_Probability", "Status"]:
    if c in paged_df.columns:
        display_cols.append(c)
for c in [f"V{i}" for i in range(1, 11)]:
    if c in paged_df.columns:
        display_cols.append(c)

st.dataframe(paged_df[display_cols], use_container_width=True)
st.caption(f"Showing rows {start_idx + 1} to {min(end_idx, len(filtered_df))} of {len(filtered_df):,} matching records.")

# CSV Download
csv_filtered = filtered_df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 Export Filtered Records as CSV",
    data=csv_filtered,
    file_name="finguard_filtered_transactions.csv",
    mime="text/csv",
    use_container_width=True
)

render_app_footer()
