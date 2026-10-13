"""
FinGuard AI - About Project & Academic Viva Guide (Page 7)
Complete documentation, system architecture, methodology, and student contribution log.
"""

import streamlit as st
from src.ui_components import apply_fintech_theme, render_sidebar_status, render_app_footer

st.set_page_config(
    page_title="About Project — FinGuard AI",
    page_icon="🛡️",
    layout="wide"
)

apply_fintech_theme()
render_sidebar_status()

# Header
st.markdown("""
<div class="fintech-header">
    <h1 class="fintech-title">📘 Project Documentation & Academic Guide</h1>
    <div class="fintech-subtitle">
        Third-Year Computer Science Diploma Semester Mini-Project &bull; FinGuard AI System Specification
    </div>
</div>
""", unsafe_allow_html=True)

# Word Document (.docx) Download Button
docx_path = config.DOCS_DIR / "FinGuard_AI_Project_Summary_and_Viva_Guide.docx"
if docx_path.exists():
    with open(docx_path, "rb") as f:
        docx_bytes = f.read()
    st.download_button(
        label="📄 Download Complete Viva Guide & Project Summary (.DOCX Word Document)",
        data=docx_bytes,
        file_name="FinGuard_AI_Project_Summary_and_Viva_Guide.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        type="primary",
        use_container_width=True
    )
    st.markdown("<div style='margin-bottom: 15px;'></div>", unsafe_allow_html=True)

# Navigation tabs for comprehensive academic report
tab_overview, tab_arch, tab_algo, tab_team, tab_ethics = st.tabs([
    "🎯 Project Overview",
    "🏗️ Architecture & Data Flow",
    "⚙️ Algorithms & Evaluation",
    "👥 Team Contributions",
    "⚖️ Limitations & Ethics"
])

# ----------------- TAB 1: OVERVIEW -----------------
with tab_overview:
    st.markdown("""
    ### 1. Problem Statement
    In modern digital economies, credit card transaction volumes have surged exponentially. Concurrently, credit card fraud causes billions of dollars in annual losses worldwide. Detecting fraudulent transactions presents a formidable machine learning challenge because:
    1. **Extreme Class Imbalance:** Legitimate transactions account for over 99.8% of daily activity, with fraud constituting less than 0.2%.
    2. **Low Latency Demands:** Financial authorization systems must evaluate transactions within milliseconds without creating friction for authentic cardholders.
    3. **Cost Asymmetry:** A false negative (missed fraud) leads to direct monetary loss, while a false positive (declining a legitimate purchase) causes customer dissatisfaction.

    ### 2. Project Objectives
    - **Genuine Machine Learning:** Develop, train, and compare three real supervised classification models (Logistic Regression, Decision Tree, Random Forest) without fabricating metrics.
    - **Leakage-Free Preprocessing:** Fit feature transformers solely on the training partition and test strictly on untouched test records.
    - **Imbalance-Aware Metric Evaluation:** Prioritize Precision-Recall AUC (PR-AUC) and F1-score over deceptive standard accuracy.
    - **Explainable Decision Interface:** Deliver dynamic threshold tuning, feature anomaly inspection, and multi-model benchmarking.
    - **Modular Personal Spending Analytics:** Integrate an independent spending intelligence module for personal expense tracking.
    """)

# ----------------- TAB 2: ARCHITECTURE -----------------
with tab_arch:
    st.markdown("""
    ### System Architecture & Pipeline Flow
    The FinGuard AI pipeline enforces strict modular separation between data ingestion, preprocessing, training, inference, and visualization:
    
    ```
    ┌────────────────────────────────────────────────────────┐
    │              Dataset Source (CSV Ingestion)            │
    │  - Real Kaggle (creditcard.csv) or Synthetic Demo Data │
    └───────────────────────────┬────────────────────────────┘
                                │ Schema Validation
                                ▼
    ┌────────────────────────────────────────────────────────┐
    │       Data Preprocessing & Leakage Prevention          │
    │  - Stratified 80/20 Train/Test Split (Preserves Ratio) │
    │  - RobustScaler on 'Time' & 'Amount' (Train Fit Only)   │
    │  - Passthrough on PCA Coordinates 'V1' through 'V28'   │
    │  - Optional SMOTE Synthesizer (Train Partition Only)   │
    └───────────────────────────┬────────────────────────────┘
                                │
                                ▼
    ┌────────────────────────────────────────────────────────┐
    │              Supervised Model Training                 │
    │  - Model A: Logistic Regression (Balanced Class Weight)│
    │  - Model B: Decision Tree Classifier (Max Depth = 6)   │
    │  - Model C: Random Forest Ensemble (100 Estimators)    │
    └───────────────────────────┬────────────────────────────┘
                                │ Evaluated on Untouched Test Set
                                ▼
    ┌────────────────────────────────────────────────────────┐
    │         Artifact Persistence & Serialization           │
    │  - models/*.joblib (Fitted Scaler & Model Binaries)    │
    │  - models/metadata.json (Metrics, CM, PR Curves)       │
    └───────────────────────────┬────────────────────────────┘
                                │
                                ▼
    ┌────────────────────────────────────────────────────────┐
    │               Streamlit Presentation Layer             │
    │  - Live Prediction Engine with Threshold Slider        │
    │  - Transaction Explorer with Pagination & Filters      │
    │  - Model Lab Benchmark & Retraining Suite              │
    │  - Spending Analytics & Outlier Detection              │
    └────────────────────────────────────────────────────────┘
    ```
    """)

# ----------------- TAB 3: ALGORITHMS -----------------
with tab_algo:
    st.markdown("""
    ### Supervised Classification Algorithms

    #### 1. Logistic Regression (Baseline Model)
    - **Mechanism:** Linear decision boundary using the sigmoid function:
      $$\\sigma(z) = \\frac{1}{1 + e^{-z}}$$
    - **Why used:** Fast, interpretable, establishes the baseline performance floor.
    - **Imbalance Handling:** `class_weight='balanced'` assigns penalty weights inversely proportional to class frequencies.

    #### 2. Decision Tree Classifier (Interpretable Non-linear Model)
    - **Mechanism:** Recursively partitions the feature space based on Gini Impurity reduction:
      $$Gini = 1 - \\sum p_i^2$$
    - **Why used:** Highly explainable, demonstrates non-linear decision thresholds without requiring feature linearity.
    - **Regularization:** Restricted to `max_depth=6` to prevent overfitting the training data.

    #### 3. Random Forest Classifier (Ensemble Benchmark)
    - **Mechanism:** Aggregates predictions across 100 decorrelated decision trees using bootstrap aggregating (bagging):
      $$\\hat{y} = \\text{Mode}(T_1(x), T_2(x), ..., T_B(x))$$
    - **Why used:** State-of-the-art tabular performance; robust to noise, outliers, and complex interactions between PCA components.

    ---
    ### Metric Evaluation Criteria
    - **Accuracy ($TP + TN / Total$):** Deceptive in fraud detection (a dummy model predicting all 0s achieves 99.8% accuracy).
    - **Recall ($TP / (TP + FN)$):** The proportion of fraudulent transactions caught. High recall minimizes bank losses.
    - **Precision ($TP / (TP + FP)$):** The proportion of fraud alerts that were actually fraud. High precision minimizes false friction.
    - **PR-AUC (Precision-Recall Area Under Curve):** The definitive metric for rare-event detection, measuring the precision-recall trade-off across all probability thresholds.
    """)

# ----------------- TAB 4: TEAM CONTRIBUTIONS -----------------
with tab_team:
    st.markdown("""
    ### Project Team Member Contributions (Diploma Mini-Project)
    
    | Student Role | Assigned Workstream | Deliverables & Responsibilities |
    | :--- | :--- | :--- |
    | **Student A** | Data Engineering & Machine Learning Pipeline | Dataset validation, stratified train/test split, RobustScaler pipeline, training Logistic Regression, Decision Tree, and Random Forest, SMOTE experiments, evaluation metrics. |
    | **Student B** | Frontend UI & Interactive Analytics | Streamlit multi-page architecture, custom dark fintech theme, Plotly chart integration, live threshold tuning slider, single & batch inference interfaces. |
    | **Student C** | Spending Analytics, Testing & Documentation | Personal spending module, automated Pytest test suite, sample data generation, architectural documentation, viva defense guide, and project report. |
    """)

# ----------------- TAB 5: ETHICS & LIMITATIONS -----------------
with tab_ethics:
    st.markdown("""
    ### Project Limitations & Ethical Considerations

    #### 1. Known Technical Limitations
    - **Static Dataset:** Evaluated on historical offline transaction batches rather than real-time streaming message queues (e.g. Apache Kafka).
    - **PCA Interpretability Barrier:** Raw merchant categories, cardholder locations, and behavioral biometric features cannot be recovered from anonymized PCA components $V1-V28$.
    - **Concept Drift:** Fraud patterns evolve rapidly as criminal syndicates adapt; static models require automated continuous retraining pipelines in production.

    #### 2. Ethical & Regulatory Principles
    - **Human-in-the-Loop:** Automated fraud classifications must be treated as probabilistic risk scores rather than irreversible legal indictments.
    - **Cardholder Privacy:** Compliance with financial data privacy standards (PCI-DSS, RBI Guidelines, GDPR) requires masking sensitive card identifiers.
    - **Algorithmic Fairness:** Disproportionate false-positive rejections must be audited to ensure algorithmic bias does not unfairly penalize specific demographic groups or geographies.
    """)

render_app_footer()
