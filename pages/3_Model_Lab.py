"""
FinGuard AI - Model Lab & Comparative Evaluation (Page 4)
Rigorous multi-model comparison, confusion matrices, PR curves, and retraining suite.
"""

import streamlit as st
import pandas as pd
import config
from src.ui_components import apply_fintech_theme, render_sidebar_status, render_app_footer
from src.train import train_all_models, load_training_metadata
from src.visualisations import (
    create_model_comparison_bar_chart,
    create_confusion_matrix_chart,
    create_pr_curve_chart,
    create_roc_curve_chart
)
from src.evaluation import explain_metric_tradeoff
from src.predict import clear_model_cache

st.set_page_config(
    page_title="Model Lab — FinGuard AI",
    page_icon="🛡️",
    layout="wide"
)

apply_fintech_theme()
selected_model = render_sidebar_status()

# Header
st.markdown("""
<div class="fintech-header">
    <h1 class="fintech-title">🔬 Model Lab & Performance Benchmark</h1>
    <div class="fintech-subtitle">
        Rigorous comparative evaluation of supervised classification models on untouched test partitions.
    </div>
</div>
""", unsafe_allow_html=True)

# Retraining Action Bar
st.markdown("### ⚡ Pipeline Retraining Suite")
train_col1, train_col2, train_col3 = st.columns([1.5, 1, 1.5])

with train_col1:
    use_smote = st.checkbox("Enable Experimental SMOTE Oversampling (Train-Set Only)", value=False, help="Synthetically augments minority fraud records strictly on the training set to help classifiers find decision boundaries.")

with train_col2:
    retrain_clicked = st.button("🚀 Retrain All Models", type="primary", use_container_width=True)

with train_col3:
    st.caption("Training applies a **Stratified 80/20 Train/Test split**. Feature scaling (RobustScaler) is fitted strictly on training data to strictly prevent data leakage.")

# Execute retraining if requested
if retrain_clicked:
    progress_bar = st.progress(0)
    status_text = st.empty()

    def update_progress(pct, text):
        progress_bar.progress(pct)
        status_text.text(text)

    try:
        current_df = st.session_state.get("current_dataset")
        meta = train_all_models(df=current_df, use_smote=use_smote, progress_callback=update_progress)
        clear_model_cache()
        st.success("All 3 models trained, evaluated, and persisted successfully!")
        st.rerun()
    except Exception as e:
        st.error(f"Training failed: {str(e)}")

# Load evaluation metadata
metadata = load_training_metadata()
if not metadata:
    st.warning("No trained models found on disk. Please click 'Retrain All Models' above to initialize the pipeline.")
    st.stop()

models_data = metadata["models"]

# Key Comparison Metrics Cards
st.markdown("---")
st.markdown("### 📊 Model Benchmark Summary Table")

table_rows = []
for m_key, m_eval in models_data.items():
    table_rows.append({
        "Model": m_eval.get("model_name", m_key),
        "Accuracy": f"{m_eval['accuracy'] * 100:.2f}%",
        "Precision": f"{m_eval['precision'] * 100:.2f}%",
        "Recall (Detection Rate)": f"{m_eval['recall'] * 100:.2f}%",
        "F1-Score": f"{m_eval['f1_score']:.4f}",
        "ROC-AUC": f"{m_eval['roc_auc']:.4f}",
        "PR-AUC (Avg Precision)": f"{m_eval['pr_auc']:.4f}",
        "Train Time (s)": f"{m_eval.get('training_time_seconds', 0.0):.2f}s"
    })

summary_df = pd.DataFrame(table_rows)
st.dataframe(summary_df, use_container_width=True)

# Recommended Model Callout
rec_key = metadata.get("recommended_model", "random_forest")
rec_name = models_data.get(rec_key, {}).get("model_name", rec_key)
st.info(f"🏆 **Recommended Model: `{rec_name}`** &bull; Selected based on optimal balance of **PR-AUC** and **F1-Score**, maximizing caught fraud while minimizing false alarms on cardholders.")

# Grouped Bar Chart Comparison
st.markdown("### 📈 Multi-Metric Comparative Visualizations")
fig_bar = create_model_comparison_bar_chart(models_data)
st.plotly_chart(fig_bar, use_container_width=True)

# Deep Dive on Selected Model
st.markdown("---")
st.markdown(f"### 🎯 In-Depth Diagnostics: `{models_data.get(selected_model, {}).get('model_name', selected_model)}`")

cur_eval = models_data.get(selected_model)
if cur_eval:
    d1, d2 = st.columns(2)

    with d1:
        # Confusion Matrix Heatmap
        fig_cm = create_confusion_matrix_chart(cur_eval["confusion_matrix"], cur_eval["model_name"])
        st.plotly_chart(fig_cm, use_container_width=True)

    with d2:
        # Precision-Recall Curve
        fig_pr = create_pr_curve_chart(cur_eval["curves"], cur_eval["model_name"], cur_eval["pr_auc"])
        st.plotly_chart(fig_pr, use_container_width=True)

    # Metric Tradeoff Narrative
    tradeoff = explain_metric_tradeoff(cur_eval)
    st.markdown("#### Operational Tradeoff Analysis")
    t_c1, t_c2 = st.columns(2)
    with t_c1:
        st.error(f"🔴 **False Negatives (Missed Fraud):** {tradeoff['false_negatives_impact']}")
    with t_c2:
        st.warning(f"🟡 **False Positives (Cardholder Friction):** {tradeoff['false_positives_impact']}")

    st.caption(f"ℹ️ {tradeoff['pr_vs_roc_note']}")

# Academic Viva Explanation Accordion
with st.expander("📚 Diploma Viva Insight: Why is Accuracy Deceptive for Fraud Detection?"):
    st.markdown("""
    **The Accuracy Paradox in Imbalanced Classification:**
    - In real credit card datasets, **99.8% of transactions are legitimate**, and only **0.2% are fraudulent**.
    - A trivial or "dumb" model that blindly predicts *every* transaction as legitimate would achieve an astonishing **99.8% accuracy**.
    - However, such a model is completely useless in production because it catches **zero fraud** (Recall = 0%).
    - For this reason, professional fraud teams and academic examiners prioritize:
      1. **Recall:** Percentage of actual fraudulent transactions detected ($TP / (TP + FN)$).
      2. **Precision:** Percentage of flagged transactions that were actually fraudulent ($TP / (TP + FP)$).
      3. **PR-AUC (Precision-Recall Area Under Curve):** Evaluates precision across all decision thresholds, focusing strictly on the minority class.
    """)

render_app_footer()
