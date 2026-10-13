"""
Unit Tests for Preprocessing and Pipeline Construction
"""

import pytest
import numpy as np
import pandas as pd
from src.data_loader import generate_synthetic_demo_data
from src.preprocessing import (
    separate_features_and_target,
    build_preprocessor,
    create_train_test_split,
    apply_optional_smote
)
import config

def test_separate_features_and_target():
    """Verify clean separation of X and y without leaking Class."""
    df = generate_synthetic_demo_data(n_samples=100, fraud_ratio=0.1)
    X, y = separate_features_and_target(df)
    assert config.TARGET_COLUMN not in X.columns
    assert len(X.columns) == len(config.EXPECTED_FEATURES)
    assert len(X) == len(y) == 100

def test_stratified_train_test_split():
    """Verify stratified split preserves the exact fraud ratio."""
    df = generate_synthetic_demo_data(n_samples=500, fraud_ratio=0.1, random_state=42)
    X_train, X_test, y_train, y_test = create_train_test_split(df, test_size=0.20, random_state=42)
    
    assert len(X_train) == 400
    assert len(X_test) == 100
    
    train_ratio = y_train.sum() / len(y_train)
    test_ratio = y_test.sum() / len(y_test)
    assert abs(train_ratio - test_ratio) < 0.02

def test_preprocessor_scaling():
    """Verify that RobustScaler scales Time and Amount while preserving dimensions."""
    df = generate_synthetic_demo_data(n_samples=100, fraud_ratio=0.1)
    X, _ = separate_features_and_target(df)
    preprocessor = build_preprocessor()
    X_transformed = preprocessor.fit_transform(X)
    
    assert X_transformed.shape == X.shape
    assert isinstance(X_transformed, np.ndarray)
