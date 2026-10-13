"""
FinGuard AI - Inference Engine
Performs model loading, schema alignment, fraud probability scoring,
and threshold-based classification.
"""

from typing import Dict, Any, Tuple, Optional, Union
import numpy as np
import pandas as pd
import joblib

import config
from src.train import MODEL_FILENAMES

# Cache loaded models in memory to optimize inference speed
_LOADED_MODELS: Dict[str, Any] = {}
_LOADED_PREPROCESSOR = None

def load_inference_artifacts(model_key: str = "random_forest") -> Tuple[Any, Any]:
    """
    Loads the trained model and associated preprocessor pipeline from disk.
    Caches artifacts to prevent redundant disk I/O.
    """
    global _LOADED_PREPROCESSOR, _LOADED_MODELS

    preprocessor_path = config.MODELS_DIR / MODEL_FILENAMES["preprocessor"]
    if not preprocessor_path.exists():
        raise FileNotFoundError(
            f"Trained preprocessor not found at {preprocessor_path}. "
            "Please train the models first using the Model Lab or python -m src.train"
        )

    if model_key not in MODEL_FILENAMES:
        raise ValueError(f"Invalid model key '{model_key}'. Supported: {list(MODEL_FILENAMES.keys())}")

    model_path = config.MODELS_DIR / MODEL_FILENAMES[model_key]
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model '{model_key}' not found at {model_path}. "
            "Please train the models first."
        )

    if _LOADED_PREPROCESSOR is None:
        _LOADED_PREPROCESSOR = joblib.load(preprocessor_path)

    if model_key not in _LOADED_MODELS:
        _LOADED_MODELS[model_key] = joblib.load(model_path)

    return _LOADED_PREPROCESSOR, _LOADED_MODELS[model_key]


def clear_model_cache():
    """Clears in-memory model cache after retraining."""
    global _LOADED_MODELS, _LOADED_PREPROCESSOR
    _LOADED_MODELS.clear()
    _LOADED_PREPROCESSOR = None


def predict_single_transaction(
    transaction_dict: Dict[str, float],
    model_key: str = "random_forest",
    threshold: float = config.DEFAULT_THRESHOLD
) -> Dict[str, Any]:
    """
    Predicts fraud status for a single transaction dictionary.
    """
    # Verify required keys
    missing = [c for c in config.EXPECTED_FEATURES if c not in transaction_dict]
    if missing:
        raise ValueError(f"Missing required transaction features: {missing[:5]}")

    # Create 1-row DataFrame with strict column order
    df_single = pd.DataFrame([transaction_dict])[config.EXPECTED_FEATURES]

    # Preprocess and predict
    preprocessor, model = load_inference_artifacts(model_key)
    X_scaled = preprocessor.transform(df_single)

    if hasattr(model, "predict_proba"):
        fraud_proba = float(model.predict_proba(X_scaled)[0, 1])
    else:
        raw_pred = float(model.predict(X_scaled)[0])
        fraud_proba = raw_pred

    is_fraud = int(fraud_proba >= threshold)
    risk_level = "High Risk" if fraud_proba >= 0.70 else ("Medium Risk" if fraud_proba >= threshold else "Low Risk")

    return {
        "model_used": model_key,
        "fraud_probability": round(fraud_proba, 4),
        "threshold": threshold,
        "is_fraud": is_fraud,
        "verdict": "Potentially Suspicious" if is_fraud == 1 else "Likely Legitimate",
        "risk_level": risk_level,
        "input_features": transaction_dict
    }


def predict_batch_transactions(
    df: pd.DataFrame,
    model_key: str = "random_forest",
    threshold: float = config.DEFAULT_THRESHOLD
) -> pd.DataFrame:
    """
    Predicts fraud status for a batch DataFrame of transactions.
    Returns annotated DataFrame with Fraud_Probability and Predicted_Class.
    """
    missing = [c for c in config.EXPECTED_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Input batch is missing {len(missing)} required features: {missing[:5]}")

    # Use strictly the feature columns in the exact trained order
    df_features = df[config.EXPECTED_FEATURES].copy()

    preprocessor, model = load_inference_artifacts(model_key)
    X_scaled = preprocessor.transform(df_features)

    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(X_scaled)[:, 1]
    else:
        probas = model.predict(X_scaled).astype(float)

    preds = (probas >= threshold).astype(int)

    result_df = df.copy()
    result_df["Fraud_Probability"] = np.round(probas, 4)
    result_df["Predicted_Class"] = preds
    result_df["Status"] = np.where(preds == 1, "Potentially Suspicious", "Likely Legitimate")

    return result_df
