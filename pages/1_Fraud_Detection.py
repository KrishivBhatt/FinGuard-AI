"""
FinGuard AI - Fraud Detection Engine (Page 2)
Interactive single-transaction and batch inference with dynamic threshold tuning.
"""

import streamlit as st
import pandas as pd
import numpy as np
import config
from src.ui_components import apply_fintech_theme, render_sidebar_status, render_app_footer
from src.predict import (
    predict_single_transaction,
    predict_batch_transactions,
    load_inference_artifacts
)
from src.data_loader import validate_fraud_dataset, generate_synthetic_demo_data

st.set_page_config(
    page_title="Fraud Detection — FinGuard AI",
    page_icon="🛡️",
    layout="wide"
)

apply_fintech_theme()
selected_model = render_sidebar_status()

# Header
st.markdown("""
<div class="fintech-header">
    <h1 class="fintech-title">🔍 Transaction Fraud Detection Engine</h1>
    <div class="fintech-subtitle">
        Score financial transactions against genuine trained Scikit-learn models with dynamic risk threshold tuning.
    </div>
</div>
""", unsafe_allow_html=True)

# Risk Disclaimer Callout
st.caption("⚠️ **Regulatory & Operational Notice:** Machine learning predictions represent probabilistic risk estimates based on mathematical patterns in historical transaction features. Predictions should be used as decision-support alerts, not absolute proof of fraud.")

# Threshold Tuning Controls
st.markdown("### ⚙️ Inference Settings & Decision Threshold")
col_thr1, col_thr2 = st.columns([2, 1])

with col_thr1:
    threshold = st.slider(
        "Classification Decision Threshold:",
        min_value=0.05,
        max_value=0.95,
        value=config.DEFAULT_THRESHOLD,
        step=0.01,
        help="Transactions with Fraud Probability >= Threshold are classified as Potentially Suspicious. Lower thresholds catch more fraud (higher recall) at the cost of more false alerts."
    )

with col_thr2:
    st.markdown(f"""
    <div class="metric-card" style="margin-top: 10px;">
        <div class="metric-label">Active Threshold</div>
        <div class="metric-value" style="color: #3B82F6;">{threshold:.2f}</div>
        <div class="metric-sub">Default: 0.50 | Lower = More Sensitive</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Inference Mode Tabs
tab_single, tab_batch = st.tabs(["🎯 Single Transaction Inspector", "📂 Batch CSV Inference"])

# ==================== TAB 1: SINGLE TRANSACTION ====================
with tab_single:
    st.subheader("Inspect & Score a Transaction")

    # Source selection
    sample_source = st.radio(
        "Transaction Data Source:",
        ["Pick from Loaded Dataset", "Load an Example Known Legitimate Transaction", "Load an Example Known Fraudulent Transaction"],
        horizontal=True
    )

    sample_dict = None
    actual_label = None

    if sample_source == "Pick from Loaded Dataset":
        current_df = st.session_state.get("current_dataset")
        if current_df is None or current_df.empty:
            st.warning("No dataset currently loaded in memory. Generating a temporary test batch...")
            current_df = generate_synthetic_demo_data(n_samples=50, fraud_ratio=0.1)
            st.session_state["current_dataset"] = current_df

        idx = st.number_input("Select Row Index:", min_value=0, max_value=len(current_df)-1, value=0, step=1)
        row = current_df.iloc[int(idx)]
        actual_label = int(row["Class"]) if "Class" in current_df.columns else None
        sample_dict = {col: float(row[col]) for col in config.EXPECTED_FEATURES if col in row}

    elif sample_source == "Load an Example Known Legitimate Transaction":
        sample_df = generate_synthetic_demo_data(n_samples=20, fraud_ratio=0.0)
        row = sample_df.iloc[0]
        actual_label = 0
        sample_dict = {col: float(row[col]) for col in config.EXPECTED_FEATURES}

    else:
        # Known Fraudulent
        sample_df = generate_synthetic_demo_data(n_samples=20, fraud_ratio=1.0)
        row = sample_df.iloc[0]
        actual_label = 1
        sample_dict = {col: float(row[col]) for col in config.EXPECTED_FEATURES}

    if sample_dict:
        # Display Transaction Details
        st.markdown("##### Key Input Attributes")
        f1, f2, f3, f4 = st.columns(4)
        f1.metric("Transaction Time", f"{sample_dict['Time']:.0f} s")
        f2.metric("Transaction Amount", f"{sample_dict['Amount']:.2f}")
        f3.metric("Key Feature V14 (PCA)", f"{sample_dict['V14']:.3f}")
        f4.metric("Actual Label (if known)", "Fraud (1)" if actual_label == 1 else ("Legitimate (0)" if actual_label == 0 else "Unknown"))

        with st.expander("🔍 View All 30 Raw Input Features (Time, V1–V28, Amount)"):
            st.json(sample_dict)

        # Run Prediction
        if st.button("🚀 Run Fraud Risk Evaluation", type="primary", use_container_width=True):
            try:
                result = predict_single_transaction(sample_dict, model_key=selected_model, threshold=threshold)
                prob = result["fraud_probability"]
                is_fraud = result["is_fraud"]

                st.markdown("### Risk Evaluation Results")
                r_col1, r_col2 = st.columns([1, 1])

                with r_col1:
                    if is_fraud == 1:
                        st.markdown(f"""
                        <div class="risk-banner risk-high">
                            🚨 VERDICT: POTENTIALLY SUSPICIOUS (FRAUD ALERT)<br>
                            <span style="font-size: 0.9rem; font-weight: 400;">
                            Estimated fraud risk exceeds the active decision threshold of {threshold:.2f}.
                            </span>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="risk-banner risk-low">
                            ✅ VERDICT: LIKELY LEGITIMATE<br>
                            <span style="font-size: 0.9rem; font-weight: 400;">
                            Estimated fraud risk is below the active decision threshold of {threshold:.2f}.
                            </span>
                        </div>
                        """, unsafe_allow_html=True)

                with r_col2:
                    st.metric(
                        label="Calculated Fraud Probability",
                        value=f"{prob * 100:.2f}%",
                        delta=f"{(prob - threshold) * 100:+.1f}% vs threshold"
                    )

                # Feature explanation note
                st.info(f"**Model Used:** `{result['model_used'].replace('_', ' ').title()}` &bull; Preprocessing applied: RobustScaler on Time & Amount; PCA coordinates passed through untouched.")

            except Exception as e:
                st.error(f"Inference error: {str(e)}")

# ==================== TAB 2: BATCH CSV INFERENCE ====================
with tab_batch:
    st.subheader("Batch Transaction Risk Scoring")
    st.markdown("Upload a CSV file containing transaction records matching the required feature schema (`Time`, `V1`–`V28`, `Amount`).")

    batch_file = st.file_uploader("Upload Batch CSV:", type=["csv"], key="batch_uploader")

    if batch_file is not None:
        try:
            batch_df = pd.read_csv(batch_file)
            is_valid, errors, warnings = validate_fraud_dataset(batch_df, require_target=False)

            if not is_valid:
                st.error("Uploaded batch file failed schema validation:\n" + "\n".join(errors))
            else:
                st.success(f"Validated {len(batch_df):,} transaction records. Schema matches all required features.")

                if st.button("⚡ Score Entire Batch", type="primary"):
                    with st.spinner("Scoring batch transactions..."):
                        annotated_df = predict_batch_transactions(
                            batch_df, model_key=selected_model, threshold=threshold
                        )

                        fraud_detected = int((annotated_df["Predicted_Class"] == 1).sum())
                        total_rows = len(annotated_df)

                        st.markdown("#### Batch Summary Telemetry")
                        b1, b2, b3 = st.columns(3)
                        b1.metric("Total Records Evaluated", f"{total_rows:,}")
                        b2.metric("Flagged as Suspicious", f"{fraud_detected:,}", delta=f"{(fraud_detected/total_rows)*100:.2f}% flagged")
                        b3.metric("Likely Legitimate", f"{total_rows - fraud_detected:,}")

                        # Preview results
                        st.markdown("##### Annotated Results Preview")
                        st.dataframe(annotated_df[["Time", "Amount", "Fraud_Probability", "Predicted_Class", "Status"] + [f"V{i}" for i in range(1, 5)]].head(100), use_container_width=True)

                        # CSV Export
                        csv_data = annotated_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download Full Scored Batch as CSV",
                            data=csv_data,
                            file_name=f"finguard_scored_batch_thr_{int(threshold*100)}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )

        except Exception as e:
            st.error(f"Failed to process batch CSV: {str(e)}")

render_app_footer()
