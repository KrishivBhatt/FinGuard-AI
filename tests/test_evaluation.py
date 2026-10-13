"""
Unit Tests for Evaluation Metrics and Tradeoff Explanations
"""

import pytest
import numpy as np
from src.evaluation import evaluate_predictions, explain_metric_tradeoff

def test_evaluate_predictions_metrics():
    """Verify that calculated metrics conform to valid mathematical ranges."""
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 1, 1, 1, 1, 0])
    y_proba = np.array([0.1, 0.2, 0.3, 0.6, 0.7, 0.8, 0.9, 0.4])

    metrics = evaluate_predictions(y_true, y_pred, y_proba)

    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert 0.0 <= metrics["f1_score"] <= 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert 0.0 <= metrics["pr_auc"] <= 1.0

    cm = metrics["confusion_matrix"]
    assert cm["tn"] == 3
    assert cm["fp"] == 1
    assert cm["fn"] == 1
    assert cm["tp"] == 3

def test_explain_metric_tradeoff():
    """Verify narrative output explaining false negatives and false positives."""
    metrics = {
        "precision": 0.85,
        "recall": 0.90,
        "confusion_matrix": {"tn": 990, "fp": 5, "fn": 2, "tp": 18}
    }
    explanation = explain_metric_tradeoff(metrics)
    assert "summary" in explanation
    assert "2 fraudulent transactions slipped through" in explanation["false_negatives_impact"]
    assert "5 legitimate transactions were falsely flagged" in explanation["false_positives_impact"]
