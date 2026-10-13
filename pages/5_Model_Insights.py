"""
FinGuard AI - Model Insights & Feature Importance (Page 6)
Explainability analysis, Gini importance, and PCA component interpretation.
"""

import streamlit as st
import numpy as np
import pandas as pd
import config
from src.ui_components import apply_fintech_theme, render_sidebar_status, render_app_footer
from src.predict import load_inference_artifacts
from src.visualisations import create_feature_importance_chart

st.set_page_config(
    page_title="Model Insights — FinGuard AI",
    page_icon="🛡️",
    layout="wide"
)

apply_fintech_theme()
selected_model = render_sidebar_status()

# Header
st.markdown("""
<div class="fintech-header">
    <h1 class="fintech-title">🧠 Model Insights & Explainability</h1>
    <div class="fintech-subtitle">
        Understand algorithmic decision drivers, feature contributions, and statistical boundaries of PCA variables.
    </div>
</div>
""", unsafe_allow_html=True)

# Important Academic Disclaimer
st.warning("""
⚠️ **Crucial Interpretability Notice for Diploma Viva:**
Features `V1` through `V28` are anonymized numerical principal components produced by **Principal Component Analysis (PCA)** to protect cardholder privacy.
**Never attempt to invent semantic real-world meanings** (e.g. claiming V1 is merchant name, or V14 is cardholder age). They are orthogonal linear combinations of original banking features.
""")

try:
    preprocessor, model = load_inference_artifacts(selected_model)
    has_importance = hasattr(model, "feature_importances_")

    if not has_importance:
        st.info(f"The active model `{selected_model.replace('_', ' ').title()}` is a linear classifier (Logistic Regression) without tree-based feature importances. To view Gini feature importance charts, select **Random Forest** or **Decision Tree** from the sidebar.")
    else:
        # Preprocessor feature order: Time, Amount, V1..V28
        feature_names = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]

        st.markdown(f"### 📊 Gini Feature Importance: `{selected_model.replace('_', ' ').title()}`")
        fig_feat = create_feature_importance_chart(model, feature_names, top_n=12)
        if fig_feat:
            st.plotly_chart(fig_feat, use_container_width=True)

        st.markdown("### 🔍 Key Interpretability Takeaways")
        i1, i2, i3 = st.columns(3)

        with i1:
            st.markdown("""
            #### 1. Dominant PCA Coordinates
            In credit card fraud detection benchmarks, components like **V14, V12, V10, V17, and V4** consistently rank highest in decision trees. In genuine datasets, these components reflect significant statistical divergence between normal cardholder behavior and stolen card anomalies.
            """)

        with i2:
            st.markdown("""
            #### 2. Transaction Amount vs Features
            While `Amount` is an intuitive financial metric, it often exhibits lower pure importance than certain PCA coordinates. This occurs because fraudsters intentionally execute low-dollar transactions to probe stolen cards without triggering automated velocity locks.
            """)

        with i3:
            st.markdown("""
            #### 3. Correlation vs Causation
            High feature importance simply indicates that a variable provided strong information gain when splitting tree nodes. It does **not prove causality**—an unusual V14 score correlates with fraud patterns, but does not cause fraud.
            """)

except Exception as e:
    st.error(f"Error loading model artifacts: {str(e)}")

# Theoretical Breakdown Accordion
st.markdown("---")
with st.expander("📖 Theoretical Guide: Understanding PCA & Tree Split Mechanics"):
    st.markdown("""
    ### Why Principal Component Analysis (PCA) is Applied to Credit Card Data:
    1. **Confidentiality:** Financial institutions are legally bound by GDPR, PCI-DSS, and banking privacy regulations. Raw customer data (names, locations, terminal IDs, merchant categories) cannot be made public.
    2. **Dimensionality Reduction:** PCA projects high-dimensional correlated attributes onto orthogonal axes that maximize data variance.
    3. **Preserved Features:** Only `Time` (seconds elapsed since first transaction) and `Amount` (transaction total) are left un-transformed in the Kaggle benchmark dataset.
    
    ### How Gini Importance is Calculated:
    In Scikit-learn, the importance of feature $j$ is computed as the total normalized reduction of the Gini impurity criterion brought by that feature across all split nodes in the tree ensemble:
    $$Gini(p) = 1 - \\sum_{k=0}^1 p_k^2$$
    Features that split nodes into pure subsets of legitimate or fraudulent transactions receive the highest importance weights.
    """)

render_app_footer()
