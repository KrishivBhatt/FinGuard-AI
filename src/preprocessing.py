"""
FinGuard AI - Preprocessing Module
Implements leakage-free feature scaling, stratified splitting, and optional SMOTE oversampling.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.compose import ColumnTransformer
import config

def separate_features_and_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separates feature matrix X and target vector y.
    Prevents leaking the Class column into feature representations.
    """
    if config.TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{config.TARGET_COLUMN}' not present in dataframe.")

    # Select exactly the expected features in defined order
    feature_cols = [col for col in config.EXPECTED_FEATURES if col in df.columns]
    X = df[feature_cols].copy()
    y = df[config.TARGET_COLUMN].astype(int).copy()
    return X, y


def build_preprocessor(scaler_type: str = "robust") -> ColumnTransformer:
    """
    Builds a Scikit-Learn ColumnTransformer.
    V1-V28 are already PCA transformed with zero mean;
    'Time' and 'Amount' have vastly different scales and outliers, so they require scaling.
    """
    scaler = RobustScaler() if scaler_type == "robust" else StandardScaler()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("scaler", scaler, ["Time", "Amount"])
        ],
        remainder="passthrough"  # V1 through V28 remain untouched
    )
    return preprocessor


def create_train_test_split(
    df: pd.DataFrame,
    test_size: float = config.TEST_SIZE,
    random_state: int = config.RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Performs stratified train/test split to maintain the exact class imbalance ratio.
    """
    X, y = separate_features_and_target(df)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    return X_train, X_test, y_train, y_test


def apply_optional_smote(
    X_train_transformed: np.ndarray,
    y_train: np.ndarray,
    random_state: int = config.RANDOM_STATE
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    Applies SMOTE (Synthetic Minority Over-sampling Technique) strictly to the training data.
    Never applied to testing data to prevent data leakage.
    Returns: (X_resampled, y_resampled, smote_applied_successfully)
    """
    try:
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=random_state)
        X_res, y_res = smote.fit_resample(X_train_transformed, y_train)
        return X_res, y_res, True
    except Exception as e:
        # Fallback if imbalanced-learn fails or is not available
        return X_train_transformed, y_train, False
