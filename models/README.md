# FinGuard AI - Saved Models & Pipeline Artifacts

This directory stores trained Scikit-learn pipelines and evaluation metadata:

- `preprocessor.joblib`: The fitted Scikit-learn `ColumnTransformer` (RobustScaler applied to `Time` & `Amount`, passthrough on `V1-V28`). Fitted strictly on training data.
- `logistic_regression.joblib`: The trained baseline Logistic Regression model with balanced class weighting.
- `decision_tree.joblib`: The trained Decision Tree Classifier (interpretable, max depth 6).
- `random_forest.joblib`: The trained ensemble Random Forest Classifier (100 estimators).
- `metadata.json`: Timestamped evaluation results, accuracy, recall, precision, F1-scores, PR-AUC, ROC-AUC, and confusion matrix counts evaluated on untouched test records.

Models can be retrained at any time from the **Model Lab** tab in the Streamlit application or via terminal using:
```bash
python -m src.train
```
