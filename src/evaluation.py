"""
FinGuard AI - Evaluation Module
Computes comprehensive classification metrics, confusion matrices, ROC/PR curves,
and beginner-friendly metric explanations.
"""

from typing import Dict, Any, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    roc_curve
)

def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray
) -> Dict[str, Any]:
    """
    Computes rigorous classification metrics on genuine predictions.
    """
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    # Core Metrics
    accuracy = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    # Area Under Curves
    try:
        roc_auc = float(roc_auc_score(y_true, y_proba))
    except Exception:
        roc_auc = 0.5

    try:
        pr_auc = float(average_precision_score(y_true, y_proba))
    except Exception:
        pr_auc = 0.0

    # Detailed report dict
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    # Curves for interactive plotting
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    prec_pts, rec_pts, _ = precision_recall_curve(y_true, y_proba)

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "matrix": cm.tolist()
        },
        "classification_report": report,
        "curves": {
            "roc": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
            "pr": {"precision": prec_pts.tolist(), "recall": rec_pts.tolist()}
        }
    }


def explain_metric_tradeoff(metrics: Dict[str, Any]) -> Dict[str, str]:
    """
    Produces plain-language insights tailored for diploma viva presentations.
    """
    rec = metrics["recall"]
    prec = metrics["precision"]
    fn = metrics["confusion_matrix"]["fn"]
    fp = metrics["confusion_matrix"]["fp"]

    if rec >= 0.85 and prec >= 0.80:
        summary = "Balanced Performance: Strong detection rate with manageable false alerts."
    elif rec > prec:
        summary = "Aggressive Fraud Catcher: High recall catches most fraud cases, but generates more false positives (flagged genuine cards)."
    elif prec > rec:
        summary = "Conservative / High Precision: Very few false alerts, but misses some fraudulent transactions (higher false negatives)."
    else:
        summary = "Moderate Performance: Needs tuning or ensemble enhancements."

    return {
        "summary": summary,
        "false_negatives_impact": f"{fn} fraudulent transactions slipped through undetected (financial loss to bank/cardholder).",
        "false_positives_impact": f"{fp} legitimate transactions were falsely flagged as fraud (friction for genuine cardholders).",
        "pr_vs_roc_note": "In fraud detection (class ratio < 0.2%), ROC-AUC can appear deceivingly high (~0.98+) because True Negatives dominate. PR-AUC focuses directly on the minority positive class and gives a realistic assessment."
    }
