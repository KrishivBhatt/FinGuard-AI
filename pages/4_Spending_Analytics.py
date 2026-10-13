"""
FinGuard AI - Personal Spending Analytics (Page 5)
Supplementary module for personal expense tracking, category budgeting, and anomaly flagging.
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import config
from src.ui_components import apply_fintech_theme, render_sidebar_status, render_app_footer
from src.spending_analytics import (
    validate_and_standardize_spending_df,
    compute_spending_metrics,
    get_category_breakdown,
    get_monthly_spending_trend,
    detect_spending_anomalies,
    generate_sample_spending_data
)
from src.visualisations import (
    create_spending_category_donut,
    create_monthly_trend_line_chart
)

st.set_page_config(
    page_title="Spending Analytics — FinGuard AI",
    page_icon="🛡️",
    layout="wide"
)

apply_fintech_theme()
render_sidebar_status()

# Header
st.markdown("""
<div class="fintech-header">
    <h1 class="fintech-title">💳 Personal Spending & Expense Analytics</h1>
    <div class="fintech-subtitle">
        Independent expense tracking module providing category breakdowns, monthly trajectories, and statistical anomaly detection.
    </div>
</div>
""", unsafe_allow_html=True)

# Currency selection
cur_col1, cur_col2 = st.columns([1, 4])
with cur_col1:
    currency = st.selectbox("Currency Format:", ["₹ (INR)", "$ (USD)", "€ (EUR)", "£ (GBP)"], index=0)
    symbol = currency.split()[0]

# Session state for spending data
if "spending_df" not in st.session_state:
    if config.SAMPLE_SPENDING_PATH.exists():
        try:
            raw_spend = pd.read_csv(config.SAMPLE_SPENDING_PATH)
            clean_spend, _ = validate_and_standardize_spending_df(raw_spend)
            st.session_state["spending_df"] = clean_spend
        except Exception:
            st.session_state["spending_df"] = generate_sample_spending_data()
    else:
        st.session_state["spending_df"] = generate_sample_spending_data()

# Upload or Generate Controls
st.markdown("### 📂 Ingest Spending Records")
up_c1, up_c2 = st.columns([2, 1])

with up_c1:
    spend_file = st.file_uploader("Upload Personal Spending CSV (Date, Amount, Category):", type=["csv"], key="spend_uploader")
    if spend_file is not None:
        try:
            raw_user_df = pd.read_csv(spend_file)
            st.caption(f"Detected columns: {list(raw_user_df.columns)}")

            # Column mapping if headers differ
            with st.expander("🛠️ Custom Column Mapping (if needed)"):
                map_date = st.selectbox("Date Column:", list(raw_user_df.columns), index=0)
                map_amt = st.selectbox("Amount Column:", list(raw_user_df.columns), index=min(1, len(raw_user_df.columns)-1))
                map_cat = st.selectbox("Category Column:", list(raw_user_df.columns), index=min(2, len(raw_user_df.columns)-1))

            std_df, errors = validate_and_standardize_spending_df(raw_user_df, map_date, map_amt, map_cat)
            if std_df is not None:
                st.session_state["spending_df"] = std_df
                st.success(f"Parsed {len(std_df):,} expenses successfully!")
            else:
                st.error("Validation failed:\n" + "\n".join(errors))
        except Exception as e:
            st.error(f"Error parsing spending CSV: {e}")

with up_c2:
    st.markdown("##### Quick Actions")
    if st.button("Generate Realistic Sample Spending History (150 Records)", use_container_width=True):
        st.session_state["spending_df"] = generate_sample_spending_data()
        st.rerun()

spend_data = st.session_state.get("spending_df")
if spend_data is None or spend_data.empty:
    st.warning("No spending records loaded.")
    st.stop()

# Date Range Filter
st.markdown("---")
st.markdown("### 📅 Date Filter & Analytics Scope")
d_min = spend_data["Date"].min().to_pydatetime()
d_max = spend_data["Date"].max().to_pydatetime()

f_d1, f_d2 = st.columns(2)
with f_d1:
    start_date = st.date_input("Start Date:", value=d_min, min_value=d_min, max_value=d_max)
with f_d2:
    end_date = st.date_input("End Date:", value=d_max, min_value=d_min, max_value=d_max)

filtered_spend = spend_data[
    (spend_data["Date"] >= pd.to_datetime(start_date)) &
    (spend_data["Date"] <= pd.to_datetime(end_date))
].copy()

# Summary Telemetry Cards
metrics = compute_spending_metrics(filtered_spend)
st.markdown("### 📊 Financial Overview")
m1, m2, m3, m4, m5 = st.columns(5)

m1.metric("Total Spending", f"{symbol}{metrics['total_spending']:,.2f}")
m2.metric("Transaction Count", f"{metrics['transaction_count']:,}")
m3.metric("Avg Transaction", f"{symbol}{metrics['avg_transaction']:,.2f}")
m4.metric("Largest Expense", f"{symbol}{metrics['max_transaction']:,.2f}")
m5.metric("Top Category", metrics['top_category'])

# Visualizations: Donut & Trend
st.markdown("---")
st.markdown("### 📈 Expense Distributions & Trajectories")
v1, v2 = st.columns(2)

with v1:
    category_df = get_category_breakdown(filtered_spend)
    fig_donut = create_spending_category_donut(category_df, currency_symbol=symbol)
    st.plotly_chart(fig_donut, use_container_width=True)

with v2:
    monthly_df = get_monthly_spending_trend(filtered_spend)
    fig_line = create_monthly_trend_line_chart(monthly_df, currency_symbol=symbol)
    st.plotly_chart(fig_line, use_container_width=True)

# Largest Transactions & Anomaly Detection
st.markdown("---")
a1, a2 = st.columns(2)

with a1:
    st.subheader("🏆 Top Largest Expenses")
    top_txns = filtered_spend.sort_values("Amount", ascending=False).head(8)
    top_display = top_txns[["Date", "Amount", "Category", "Description"]].copy()
    top_display["Date"] = top_display["Date"].dt.strftime("%d %b %Y")
    top_display["Amount"] = top_display["Amount"].apply(lambda v: f"{symbol}{v:,.2f}")
    st.dataframe(top_display, use_container_width=True)

with a2:
    st.subheader("⚠️ Outlier Expense Detection (Experimental)")
    st.caption("Statistical anomaly detection based on Standard Deviation (Z-score > 2.5). **Note:** Separate from the supervised credit card fraud model.")
    anomalies = detect_spending_anomalies(filtered_spend, threshold_std=2.5)
    if not anomalies.empty:
        anom_disp = anomalies[["Date", "Amount", "Category", "Description", "Status"]].copy()
        anom_disp["Date"] = anom_disp["Date"].dt.strftime("%d %b %Y")
        anom_disp["Amount"] = anom_disp["Amount"].apply(lambda v: f"{symbol}{v:,.2f}")
        st.dataframe(anom_disp, use_container_width=True)
    else:
        st.success("No statistical spending anomalies detected within the selected timeframe.")

# Export
csv_spend = filtered_spend.to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 Export Filtered Expenses as CSV",
    data=csv_spend,
    file_name="finguard_spending_records.csv",
    mime="text/csv",
    use_container_width=True
)

render_app_footer()
