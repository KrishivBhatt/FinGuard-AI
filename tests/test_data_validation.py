"""
Unit Tests for Data Validation & Schema Checks
"""

import pytest
import pandas as pd
import numpy as np
import config
from src.data_loader import validate_fraud_dataset, generate_synthetic_demo_data

def test_valid_synthetic_dataset():
    """Verify that generated demo dataset passes validation completely."""
    df = generate_synthetic_demo_data(n_samples=200, fraud_ratio=0.05, random_state=42)
    is_valid, errors, warnings = validate_fraud_dataset(df, require_target=True)
    assert is_valid is True
    assert len(errors) == 0

def test_missing_required_columns():
    """Verify error reporting when critical feature columns are missing."""
    df = pd.DataFrame({
        "Time": [0, 1],
        "V1": [0.1, -0.2],
        "Amount": [10.0, 50.0],
        "Class": [0, 1]
    })
    is_valid, errors, _ = validate_fraud_dataset(df, require_target=True)
    assert is_valid is False
    assert any("Missing" in err for err in errors)

def test_single_class_target():
    """Verify error reporting when target column contains only one class."""
    df = generate_synthetic_demo_data(n_samples=100, fraud_ratio=0.0, random_state=42)
    df["Class"] = 0  # Only legitimate
    is_valid, errors, _ = validate_fraud_dataset(df, require_target=True)
    assert is_valid is False
    assert any("only one class" in err for err in errors)

def test_invalid_target_values():
    """Verify error reporting when target contains invalid labels (e.g. 2, 3 or strings)."""
    df = generate_synthetic_demo_data(n_samples=100, fraud_ratio=0.05, random_state=42)
    df.loc[0, "Class"] = 99
    is_valid, errors, _ = validate_fraud_dataset(df, require_target=True)
    assert is_valid is False
    assert any("invalid classes" in err for err in errors)

def test_non_numeric_features():
    """Verify error reporting when features contain non-numeric data."""
    df = generate_synthetic_demo_data(n_samples=50, fraud_ratio=0.05, random_state=42)
    df["V5"] = "non-numeric-string"
    is_valid, errors, _ = validate_fraud_dataset(df, require_target=True)
    assert is_valid is False
    assert any("must be numeric" in err for err in errors)
