"""
FinGuard AI - Data Loader & Validation Module
Handles reading, validating, and generating credit card transaction data.
"""

from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import pandas as pd
from pathlib import Path
import config

def validate_fraud_dataset(df: pd.DataFrame, require_target: bool = True) -> Tuple[bool, List[str], List[str]]:
    """
    Validates a credit card dataset against the expected schema.
    Returns:
        (is_valid, errors, warnings)
    """
    errors = []
    warnings = []

    if df is None or df.empty:
        errors.append("Dataset is empty or could not be parsed.")
        return False, errors, warnings

    # Check required feature columns
    missing_features = [col for col in config.EXPECTED_FEATURES if col not in df.columns]
    if missing_features:
        errors.append(f"Missing {len(missing_features)} required feature columns: {missing_features[:5]}{'...' if len(missing_features) > 5 else ''}")

    # Check target column if required
    if require_target:
        if config.TARGET_COLUMN not in df.columns:
            errors.append(f"Missing target column '{config.TARGET_COLUMN}'.")
        else:
            unique_classes = set(df[config.TARGET_COLUMN].dropna().unique())
            # Target must contain only binary {0, 1}
            invalid_classes = unique_classes - {0, 1}
            if invalid_classes:
                errors.append(f"Target column '{config.TARGET_COLUMN}' contains invalid classes: {invalid_classes}. Expected only 0 (Legitimate) and 1 (Fraud).")
            if len(unique_classes) < 2:
                errors.append(f"Dataset target column contains only one class ({unique_classes}). Both 0 (Legitimate) and 1 (Fraud) are required to train a classifier.")

    # Check data types (all features should be numeric)
    non_numeric_cols = []
    for col in df.columns:
        if col in config.ALL_COLUMNS:
            if not pd.api.types.is_numeric_dtype(df[col]):
                non_numeric_cols.append(col)
    if non_numeric_cols:
        errors.append(f"Columns must be numeric. Found non-numeric data in: {non_numeric_cols[:5]}")

    # Check for missing values
    null_count = df.isnull().sum().sum()
    if null_count > 0:
        warnings.append(f"Dataset contains {null_count} missing (NaN) values across features.")

    # Check for duplicate rows
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        warnings.append(f"Dataset contains {duplicate_count:,} duplicate rows.")

    is_valid = len(errors) == 0
    return is_valid, errors, warnings


def load_dataset_from_path(filepath: Path) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    """
    Loads dataset from local file path with error handling.
    """
    result = {"success": False, "error": None, "source": str(filepath)}
    if not filepath.exists():
        result["error"] = f"File not found at: {filepath}"
        return None, result

    try:
        df = pd.read_csv(filepath)
        result["success"] = True
        result["rows"] = len(df)
        result["columns"] = len(df.columns)
        return df, result
    except Exception as e:
        result["error"] = f"Failed to read CSV file: {str(e)}"
        return None, result


def generate_synthetic_demo_data(n_samples: int = 5000, fraud_ratio: float = 0.015, random_state: int = 42) -> pd.DataFrame:
    """
    Generates a deterministic synthetic dataset matching the Kaggle credit card schema
    for immediate demonstration and testing when the 150MB Kaggle CSV is not yet placed.
    Clearly marked as synthetic in metadata.
    """
    rng = np.random.RandomState(random_state)
    n_fraud = max(10, int(n_samples * fraud_ratio))
    n_legit = n_samples - n_fraud

    # Generate legitimate transactions: V1-V28 roughly standard normal
    legit_v = rng.randn(n_legit, 28)
    legit_time = np.sort(rng.uniform(0, 172800, n_legit)) # 48 hours in seconds
    legit_amount = np.round(rng.exponential(scale=70.0, size=n_legit) + 1.0, 2)
    legit_class = np.zeros(n_legit, dtype=int)

    # Generate fraud transactions: noticeably shifted distributions on key PCA features (e.g., V4, V11, V12, V14, V17)
    fraud_v = rng.randn(n_fraud, 28)
    fraud_v[:, 3] += 3.5   # V4 higher in fraud
    fraud_v[:, 10] += 3.0  # V11 higher in fraud
    fraud_v[:, 11] -= 4.0  # V12 lower in fraud
    fraud_v[:, 13] -= 5.0  # V14 noticeably lower in fraud
    fraud_v[:, 16] -= 4.0  # V17 noticeably lower in fraud
    fraud_time = np.sort(rng.uniform(0, 172800, n_fraud))
    fraud_amount = np.round(rng.exponential(scale=180.0, size=n_fraud) + 10.0, 2)
    fraud_class = np.ones(n_fraud, dtype=int)

    # Combine
    all_v = np.vstack([legit_v, fraud_v])
    all_time = np.concatenate([legit_time, fraud_time])
    all_amount = np.concatenate([legit_amount, fraud_amount])
    all_class = np.concatenate([legit_class, fraud_class])

    v_cols = [f"V{i}" for i in range(1, 29)]
    df = pd.DataFrame(all_v, columns=v_cols)
    df.insert(0, "Time", all_time)
    df["Amount"] = all_amount
    df["Class"] = all_class

    # Shuffle rows deterministically
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return df


def get_dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes statistical and structural summary of the loaded dataset.
    """
    if df is None or df.empty:
        return {
            "total_records": 0,
            "legit_count": 0,
            "fraud_count": 0,
            "fraud_percentage": 0.0,
            "imbalance_ratio": "N/A",
            "features_count": 0,
            "missing_values": 0,
            "mean_amount": 0.0,
            "max_amount": 0.0,
            "min_amount": 0.0
        }

    total = len(df)
    has_target = config.TARGET_COLUMN in df.columns
    if has_target:
        class_counts = df[config.TARGET_COLUMN].value_counts().to_dict()
        legit = int(class_counts.get(0, 0))
        fraud = int(class_counts.get(1, 0))
        fraud_pct = (fraud / total * 100.0) if total > 0 else 0.0
        imbalance_str = f"1 : {round(legit / fraud, 1)}" if fraud > 0 else "No fraud instances"
    else:
        legit, fraud, fraud_pct, imbalance_str = 0, 0, 0.0, "Target not present"

    amount_series = df["Amount"] if "Amount" in df.columns else pd.Series([0.0])

    return {
        "total_records": total,
        "legit_count": legit,
        "fraud_count": fraud,
        "fraud_percentage": round(fraud_pct, 4),
        "imbalance_ratio": imbalance_str,
        "features_count": len([c for c in df.columns if c != config.TARGET_COLUMN]),
        "missing_values": int(df.isnull().sum().sum()),
        "mean_amount": round(float(amount_series.mean()), 2),
        "median_amount": round(float(amount_series.median()), 2),
        "max_amount": round(float(amount_series.max()), 2),
        "min_amount": round(float(amount_series.min()), 2),
        "has_target": has_target
    }
