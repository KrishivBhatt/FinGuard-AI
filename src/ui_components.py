"""
FinGuard AI - Reusable UI Components & Styling
Provides consistent fintech styling, status panels, and cards across all pages.
"""

import streamlit as st
from pathlib import Path
import config
from src.train import load_training_metadata

def apply_fintech_theme():
    """
    Applies custom CSS for a premium dark fintech aesthetic.
    """
    st.markdown("""
    <style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    /* Fintech Header Banner */
    .fintech-header {
        background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 50%, #1E293B 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 24px 32px;
        margin-bottom: 24px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.25);
    }
    .fintech-title {
        color: #FFFFFF;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .fintech-subtitle {
        color: #94A3B8;
        font-size: 1.05rem;
        margin-top: 6px;
        font-weight: 400;
    }

    /* Status Badge */
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .badge-success { background-color: rgba(16, 185, 129, 0.2); color: #10B981; border: 1px solid #10B981; }
    .badge-warning { background-color: rgba(245, 158, 11, 0.2); color: #F59E0B; border: 1px solid #F59E0B; }
    .badge-danger { background-color: rgba(239, 68, 68, 0.2); color: #EF4444; border: 1px solid #EF4444; }
    .badge-info { background-color: rgba(59, 130, 246, 0.2); color: #3B82F6; border: 1px solid #3B82F6; }

    /* Metric Cards */
    .metric-card {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 18px 20px;
        margin-bottom: 15px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.2);
        border-color: #3B82F6;
    }
    .metric-label {
        font-size: 0.82rem;
        color: #94A3B8;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.7rem;
        font-weight: 700;
        color: #F8FAFC;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #64748B;
        margin-top: 4px;
    }

    /* Streamlit Metric Overrides */
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        color: #F8FAFC !important;
    }

    /* Tables */
    div[data-testid="stDataFrame"] {
        border: 1px solid #334155;
        border-radius: 8px;
        overflow: hidden;
    }

    /* Risk Score Gauge Container */
    .risk-banner {
        padding: 16px 20px;
        border-radius: 8px;
        margin: 12px 0;
        font-weight: 600;
    }
    .risk-high { background-color: rgba(239, 68, 68, 0.15); border-left: 5px solid #EF4444; color: #FCA5A5; }
    .risk-medium { background-color: rgba(245, 158, 11, 0.15); border-left: 5px solid #F59E0B; color: #FCD34D; }
    .risk-low { background-color: rgba(16, 185, 129, 0.15); border-left: 5px solid #10B981; color: #6EE7B7; }

    /* Footer */
    .app-footer {
        text-align: center;
        color: #64748B;
        font-size: 0.85rem;
        margin-top: 40px;
        padding-top: 20px;
        border-top: 1px solid #334155;
    }
    </style>
    """, unsafe_allow_html=True)


def render_sidebar_status() -> str:
    """
    Renders persistent sidebar panel with dataset status, model status, and model selector.
    Returns: selected_model_key
    """
    st.sidebar.markdown("""
    <div style="text-align: center; padding-bottom: 12px; margin-bottom: 12px; border-bottom: 1px solid #334155;">
        <h2 style="margin: 0; color: #3B82F6; font-size: 1.4rem;">🛡️ FinGuard AI</h2>
        <span style="color: #94A3B8; font-size: 0.8rem;">Fintech Risk & Spending Intelligence</span>
    </div>
    """, unsafe_allow_html=True)

    metadata = load_training_metadata()

    # Model Selection Box
    st.sidebar.subheader("Active Inference Model")
    model_options = {
        "random_forest": "Random Forest (Recommended)",
        "decision_tree": "Decision Tree",
        "logistic_regression": "Logistic Regression"
    }

    default_idx = 0
    if metadata and metadata.get("recommended_model") in model_options:
        default_idx = list(model_options.keys()).index(metadata["recommended_model"])

    selected_model = st.sidebar.selectbox(
        "Choose Classification Model:",
        options=list(model_options.keys()),
        format_func=lambda k: model_options[k],
        index=default_idx
    )

    # Dataset Status Indicator
    st.sidebar.markdown("---")
    st.sidebar.subheader("System Status")

    if config.REAL_DATASET_PATH.exists():
        st.sidebar.markdown("📁 **Dataset:** `creditcard.csv` (Real)")
        st.sidebar.markdown('<span class="status-badge badge-success">Production Benchmark Loaded</span>', unsafe_allow_html=True)
    elif config.SAMPLE_DATASET_PATH.exists():
        st.sidebar.markdown("📁 **Dataset:** `sample_transactions.csv` (Demo)")
        st.sidebar.markdown('<span class="status-badge badge-warning">Synthetic Demo Mode</span>', unsafe_allow_html=True)
    else:
        st.sidebar.markdown("📁 **Dataset:** None loaded")
        st.sidebar.markdown('<span class="status-badge badge-danger">Dataset Missing</span>', unsafe_allow_html=True)

    if metadata:
        train_type = "Synthetic Demo" if metadata.get("is_synthetic") else "Real Kaggle Data"
        st.sidebar.markdown(f"🤖 **Model Engine:** Trained ({train_type})")
        st.sidebar.markdown(f"📊 **Test Samples:** {metadata.get('test_rows', 0):,}")
    else:
        st.sidebar.markdown("🤖 **Model Engine:** Not Trained")

    st.sidebar.markdown("---")
    st.sidebar.info("💡 **Academic Project:** 3rd Year Diploma CS Mini-Project. Models strictly use genuine trained Scikit-learn pipelines.")

    return selected_model


def render_app_footer():
    """Renders academic project footer."""
    st.markdown("""
    <div class="app-footer">
        <b>FinGuard AI — Machine Learning Fraud Detection & Spending Analytics</b><br>
        Third-Year Computer Science Diploma Semester Mini-Project &bull; Built with Python & Streamlit &bull; 100% Genuine ML Pipeline
    </div>
    """, unsafe_allow_html=True)
