# FinGuard AI — Technical Architecture & Pipeline Specification

This document details the architectural design, component interactions, data contracts, and data-flow lifecycle of the FinGuard AI system.

---

## 1. High-Level Architecture Diagram

```
+------------------------------------------------------------------------------------+
|                               PRESENTATION TIER                                    |
|   Streamlit Multi-Page Web App (Fintech Theme, Interactive Plotly Visualizations)  |
|                                                                                    |
|  [app.py]        [1_Fraud_Detection]  [2_Transaction_Explorer]  [3_Model_Lab]      |
|  Executive       Single & Batch       Filtering & CSV Export    Model Comparison,  |
|  Dashboard       Threshold Tuning                               Retraining Suite   |
|                                                                                    |
|  [4_Spending_Analytics]               [5_Model_Insights]        [6_About_Project]  |
|  Personal Expense Analytics           Feature Importance        Academic Report &  |
|  & Anomaly Detector                   & PCA Guide               Viva Documentation |
+------------------------------------------+-----------------------------------------+
                                           | calls
                                           v
+------------------------------------------------------------------------------------+
|                                APPLICATION LOGIC TIER                              |
|                                                                                    |
|  [src/predict.py]          [src/spending_analytics.py]   [src/visualisations.py]   |
|  Inference Engine &        Expense Aggregation,          Plotly Dark Styling,      |
|  Probability Thresholding  Z-score Outlier Detection     Heatmaps, Curves          |
|                                                                                    |
|  [src/train.py]            [src/preprocessing.py]        [src/evaluation.py]       |
|  Multi-Model Training      Stratified Split (80/20),     Metrics (PR-AUC, ROC-AUC),|
|  Orchestrator              RobustScaler ColumnTransform  Confusion Matrix          |
|                                                                                    |
|  [src/data_loader.py]      [src/ui_components.py]                                  |
|  Schema Validator &        Shared CSS Styles &                                     |
|  Synthetic Demo Generator  Persistent Status Panel                                 |
+------------------------------------------+-----------------------------------------+
                                           | reads / writes
                                           v
+------------------------------------------------------------------------------------+
|                                 PERSISTENCE TIER                                   |
|                                                                                    |
|  data/                                  models/                                    |
|  - creditcard.csv (Kaggle Benchmark)    - preprocessor.joblib                      |
|  - sample_transactions.csv (Demo Batch) - logistic_regression.joblib               |
|  - sample_spending.csv (Expenses)       - decision_tree.joblib                     |
|                                         - random_forest.joblib                     |
|                                         - metadata.json (Evaluated Metrics)        |
+------------------------------------------------------------------------------------+
```

---

## 2. Ingestion & Preprocessing Data Flow

```
[Raw CSV File]
      |
      v
(src/data_loader.py -> validate_fraud_dataset)
      |
      +---> Check: Time, V1-V28, Amount present?
      +---> Check: All columns numeric?
      +---> Check: Binary target (0 and 1)?
      |
      v
(src/preprocessing.py -> create_train_test_split)
      |
      +---> Stratified 80/20 Split
      |
      +------------------------+
      |                        |
      v (Training 80%)         v (Untouched Testing 20%)
[X_train, y_train]        [X_test, y_test]
      |                        |
      v                        |
[Fit RobustScaler on           |
 Time & Amount, Passthrough    |
 on V1-V28]                    |
      |                        |
      +-------> [Transform X_train]
      |                        |
      +------------------------+-------> [Transform X_test using
      |                                   FITTED Training Scaler]
      v
(Optional: apply_optional_smote on X_train only)
      |
      v
[Train Classifiers: LR, DT, RF]
      |
      v
[Predict on X_test]
      |
      v
(src/evaluation.py -> evaluate_predictions)
      |
      v
[Save .joblib Models & metadata.json]
```

---

## 3. Inference Engine Data Flow

When a user submits a single transaction or uploads a batch CSV in `1_Fraud_Detection.py`:

```
User Input (JSON Dict or CSV DataFrame)
      |
      v
(src/predict.py -> validate feature columns)
      |
      v
Load Cached Preprocessor (`models/preprocessor.joblib`)
      |
      v
Scale `Time` & `Amount` using training parameters; passthrough `V1-V28`
      |
      v
Load Cached Model (`models/random_forest.joblib`)
      |
      v
Execute `model.predict_proba(X_scaled)[:, 1]`
      |
      v
Calculate: `is_fraud = int(probability >= threshold)`
      |
      v
Return Risk Dictionary or Annotated DataFrame (Fraud_Probability, Predicted_Class, Status)
```

---

## 4. Key Architectural Safeguards

1. **Strict Data Leakage Prevention:**
   The preprocessor (`ColumnTransformer`) fits its scaling parameters (median and IQR) exclusively on `X_train`. `X_test` is transformed without re-fitting. This guarantees that test evaluation reflects realistic generalization on unseen data.
2. **Deterministic Reproducibility:**
   All splitting and model initializations use fixed random states (`config.RANDOM_STATE = 42`).
3. **In-Memory Caching:**
   Inference pipelines in `src/predict.py` maintain global module caches for loaded `.joblib` objects, preventing repetitive disk I/O on rapid prediction calls.
4. **Resilient Decoupling:**
   The Personal Spending module operates independently of the PCA fraud model, ensuring that failure in one analytical domain does not degrade the core classification system.
