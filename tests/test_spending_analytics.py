"""
Unit Tests for Spending Analytics and Anomaly Detection
"""

import pytest
import pandas as pd
from src.spending_analytics import (
    validate_and_standardize_spending_df,
    compute_spending_metrics,
    get_category_breakdown,
    get_monthly_spending_trend,
    detect_spending_anomalies,
    generate_sample_spending_data
)

def test_generate_and_standardize_spending():
    """Verify sample generation and standardization."""
    sample_df = generate_sample_spending_data(50)
    clean_df, errors = validate_and_standardize_spending_df(sample_df)
    assert len(errors) == 0
    assert clean_df is not None
    assert set(["Date", "Amount", "Category", "Description"]).issubset(clean_df.columns)

def test_compute_spending_metrics():
    """Verify aggregated metrics for spending data."""
    df = pd.DataFrame({
        "Date": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"]),
        "Amount": [100.0, 200.0, 300.0],
        "Category": ["Food", "Transport", "Food"],
        "Description": ["A", "B", "C"]
    })
    metrics = compute_spending_metrics(df)
    assert metrics["total_spending"] == 600.0
    assert metrics["transaction_count"] == 3
    assert metrics["avg_transaction"] == 200.0
    assert metrics["top_category"] == "Food"

def test_category_breakdown():
    """Verify category group totals and percentages."""
    df = pd.DataFrame({
        "Date": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"]),
        "Amount": [100.0, 300.0, 100.0],
        "Category": ["Food", "Shopping", "Food"]
    })
    breakdown = get_category_breakdown(df)
    assert len(breakdown) == 2
    assert breakdown.iloc[0]["Category"] == "Shopping"
    assert breakdown.iloc[0]["Percentage"] == 60.0

def test_detect_spending_anomalies():
    """Verify Z-score outlier detection."""
    df = pd.DataFrame({
        "Date": pd.to_datetime([f"2025-01-{i:02d}" for i in range(1, 20)]),
        "Amount": [50.0] * 18 + [50000.0],  # 1 massive outlier
        "Category": ["General"] * 19,
        "Description": ["Normal"] * 18 + ["Extreme Outlier"]
    })
    anomalies = detect_spending_anomalies(df, threshold_std=2.0)
    assert len(anomalies) >= 1
    assert anomalies.iloc[0]["Amount"] == 50000.0
