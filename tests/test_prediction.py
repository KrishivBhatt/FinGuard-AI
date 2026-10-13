"""
Unit Tests for Model Prediction and Thresholding
"""

import pytest
import numpy as np
import pandas as pd
from src.data_loader import generate_synthetic_demo_data
from src.predict import (
    predict_single_transaction,
    predict_batch_transactions,
    load_inference_artifacts
)
import config

def test_inference_artifacts_load():
    """Verify that saved preprocessor and model load correctly."""
    preprocessor, model = load_inference_artifacts("random_forest")
    assert preprocessor is not None
    assert model is not None

def test_single_prediction_format_and_threshold():
    """Verify single transaction prediction outputs correct types and obeys threshold."""
    df = generate_synthetic_demo_data(n_samples=20, fraud_ratio=0.2, random_state=42)
    sample_txn = df.drop(columns=["Class"]).iloc[0].to_dict()
    
    # Predict with default threshold 0.5
    res = predict_single_transaction(sample_txn, model_key="random_forest", threshold=0.5)
    assert "fraud_probability" in res
    assert "is_fraud" in res
    assert 0.0 <= res["fraud_probability"] <= 1.0
    assert res["is_fraud"] in [0, 1]
    
    # Test strict threshold 0.01 vs 0.99
    res_low = predict_single_transaction(sample_txn, threshold=0.0001)
    res_high = predict_single_transaction(sample_txn, threshold=0.9999)
    # The classification should mathematically depend on threshold
    if res["fraud_probability"] > 0.0001:
        assert res_low["is_fraud"] == 1
    if res["fraud_probability"] < 0.9999:
        assert res_high["is_fraud"] == 0

def test_batch_prediction():
    """Verify batch inference adds Fraud_Probability and Predicted_Class columns."""
    df = generate_synthetic_demo_data(n_samples=50, fraud_ratio=0.1, random_state=42)
    features_df = df.drop(columns=["Class"])
    
    annotated = predict_batch_transactions(features_df, model_key="random_forest", threshold=0.5)
    assert "Fraud_Probability" in annotated.columns
    assert "Predicted_Class" in annotated.columns
    assert "Status" in annotated.columns
    assert len(annotated) == 50
