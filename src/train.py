"""
FinGuard AI - Model Training Pipeline
Trains Logistic Regression, Decision Tree, and Random Forest models on credit card transactions.
Saves preprocessors, trained models, and performance metrics.
"""

import time
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import config
from src.data_loader import (
    load_dataset_from_path,
    generate_synthetic_demo_data,
    validate_fraud_dataset
)
from src.preprocessing import (
    create_train_test_split,
    build_preprocessor,
    apply_optional_smote
)
from src.evaluation import evaluate_predictions

MODEL_FILENAMES = {
    "logistic_regression": "logistic_regression.joblib",
    "decision_tree": "decision_tree.joblib",
    "random_forest": "random_forest.joblib",
    "preprocessor": "preprocessor.joblib"
}

METADATA_FILE = config.MODELS_DIR / "metadata.json"


def train_all_models(
    df: Optional[pd.DataFrame] = None,
    use_smote: bool = False,
    progress_callback: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Trains all 3 classification models, calculates performance on held-out test data,
    and persists models and artifacts to disk.
    """
    is_synthetic = False
    if df is None:
        if config.REAL_DATASET_PATH.exists():
            df, _ = load_dataset_from_path(config.REAL_DATASET_PATH)
        else:
            # Generate synthetic demo data to ensure zero-crash hands-on experience
            df = generate_synthetic_demo_data(n_samples=6000, fraud_ratio=0.017)
            is_synthetic = True

    is_valid, errors, _ = validate_fraud_dataset(df, require_target=True)
    if not is_valid:
        raise ValueError(f"Dataset validation failed: {'; '.join(errors)}")

    if progress_callback:
        progress_callback(10, "Splitting dataset into stratified train & test sets...")

    # 1. Stratified Train/Test Split (80/20)
    X_train, X_test, y_train, y_test = create_train_test_split(df)

    if progress_callback:
        progress_callback(25, "Fitting preprocessor on training data only...")

    # 2. Fit Preprocessor ONLY on training data (Strictly avoid data leakage)
    preprocessor = build_preprocessor()
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)

    # 3. Handle Imbalance (SMOTE if requested, else class_weight='balanced')
    smote_used = False
    if use_smote:
        if progress_callback:
            progress_callback(35, "Applying SMOTE oversampling to training set...")
        X_train_resampled, y_train_resampled, smote_used = apply_optional_smote(
            X_train_scaled, y_train.values
        )
    else:
        X_train_resampled, y_train_resampled = X_train_scaled, y_train.values

    # Model Definitions
    class_weight = None if smote_used else "balanced"
    models_to_train = {
        "logistic_regression": {
            "name": "Logistic Regression",
            "model": LogisticRegression(
                max_iter=1000,
                class_weight=class_weight,
                random_state=config.RANDOM_STATE
            )
        },
        "decision_tree": {
            "name": "Decision Tree",
            "model": DecisionTreeClassifier(
                max_depth=6,
                class_weight=class_weight,
                random_state=config.RANDOM_STATE
            )
        },
        "random_forest": {
            "name": "Random Forest",
            "model": RandomForestClassifier(
                n_estimators=100,
                max_depth=12,
                class_weight=class_weight,
                random_state=config.RANDOM_STATE,
                n_jobs=-1
            )
        }
    }

    results = {}
    total_models = len(models_to_train)

    for idx, (m_key, m_info) in enumerate(models_to_train.items()):
        step_pct = 40 + int((idx / total_models) * 50)
        if progress_callback:
            progress_callback(step_pct, f"Training {m_info['name']}...")

        start_time = time.time()
        clf = m_info["model"]
        clf.fit(X_train_resampled, y_train_resampled)
        duration = round(time.time() - start_time, 3)

        # Predictions on UNTOUCHED held-out test data
        y_pred = clf.predict(X_test_scaled)
        y_proba = clf.predict_proba(X_test_scaled)[:, 1] if hasattr(clf, "predict_proba") else y_pred.astype(float)

        # Calculate metrics
        metrics = evaluate_predictions(y_test.values, y_pred, y_proba)
        metrics["training_time_seconds"] = duration
        metrics["model_name"] = m_info["name"]

        # Save individual model
        model_path = config.MODELS_DIR / MODEL_FILENAMES[m_key]
        joblib.dump(clf, model_path)

        results[m_key] = metrics

    if progress_callback:
        progress_callback(95, "Saving preprocessor and metadata...")

    # Save Preprocessor
    joblib.dump(preprocessor, config.MODELS_DIR / MODEL_FILENAMES["preprocessor"])

    # Determine recommended model based on PR-AUC and F1-Score (Standard for fraud)
    best_model_key = max(results.keys(), key=lambda k: (results[k]["pr_auc"] * 0.6 + results[k]["f1_score"] * 0.4))

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "is_synthetic": is_synthetic,
        "smote_applied": smote_used,
        "dataset_rows": len(df),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "fraud_in_test": int(y_test.sum()),
        "recommended_model": best_model_key,
        "models": results
    }

    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=2)

    if progress_callback:
        progress_callback(100, "Model training complete!")

    return metadata


def load_training_metadata() -> Optional[Dict[str, Any]]:
    """
    Loads saved metadata.json if available.
    """
    if METADATA_FILE.exists():
        try:
            with open(METADATA_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return None
    return None


if __name__ == "__main__":
    print("Executing FinGuard AI model training pipeline...")
    meta = train_all_models()
    print("Training finished successfully!")
    print(f"Recommended Model: {meta['recommended_model']}")
    for k, v in meta["models"].items():
        print(f"- {v['model_name']}: Accuracy={v['accuracy']}, Recall={v['recall']}, Precision={v['precision']}, PR-AUC={v['pr_auc']}")
