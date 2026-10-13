# FinGuard AI — Intelligent Credit Card Fraud Detection & Spending Analytics

![FinGuard AI Banner](https://img.shields.io/badge/FinGuard%20AI-Fintech%20Risk%20Analytics-1E3A8A?style=for-the-badge)
![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3B82F6?style=flat-square)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B?style=flat-square)
![Scikit-Learn](https://img.shields.io/badge/ML%20Engine-Scikit--Learn-F59E0B?style=flat-square)
![Tests](https://img.shields.io/badge/Tests-17%20Passed-10B981?style=flat-square)

> **Third-Year Computer Science Diploma Semester Mini-Project**  
> Developed as a production-grade machine learning application designed to evaluate financial transaction risks, benchmark classification algorithms, and track personal spending habits.

---

## 📌 1. Project Overview & Problem Statement
In modern digital financial systems, payment cards process trillions of dollars annually. Fraudulent transactions result in billions in losses and erode cardholder confidence. Detecting financial fraud presents distinct engineering challenges:
1. **Severe Class Imbalance:** Fraudulent transactions typically represent less than **0.2%** of total card volume. Standard accuracy is deceptive and clinically useless.
2. **Data Privacy Barriers:** Financial transaction datasets must be anonymized (using Principal Component Analysis) to adhere to strict international privacy mandates (PCI-DSS, GDPR).
3. **Operational Trade-offs:** Striking the optimal balance between **Recall** (catching fraudulent attempts) and **Precision** (avoiding false alerts on authentic cardholders).

**FinGuard AI** solves this by establishing a rigorous, end-to-end Machine Learning pipeline comparing **Logistic Regression**, **Decision Trees**, and **Random Forest Classifiers**, coupled with dynamic thresholding and spending analytics.

---

## 🌟 2. Key Features

- **Genuine Machine Learning Pipelines:**
  - Zero fabricated metrics; all probabilities originate from genuine Scikit-learn pipelines.
  - Strict **data leakage prevention**: Feature scaling (`RobustScaler`) is fitted strictly on training data.
  - Stratified 80/20 train/test splitting preserving real-world fraud imbalance ratios.
  - Optional **SMOTE** (Synthetic Minority Over-sampling Technique) applied strictly to training partitions.
- **Dynamic Risk Threshold Tuning:**
  - Adjust the classification threshold from 0.05 to 0.95 with instantaneous classification updates.
- **Single & Batch Transaction Inference:**
  - Score individual transactions with feature inspection and risk gauge indicators.
  - Upload CSV batches, validate schemas, score in bulk, and export annotated results.
- **Searchable Transaction Explorer:**
  - Filter transactions by amount, actual class, model prediction, or minimum probability with pagination and CSV export.
- **Model Lab & Comparative Benchmark:**
  - Interactive multi-metric bar charts, confusion matrix heatmaps, ROC curves, and Precision-Recall (PR) curves.
- **Personal Spending Analytics Module:**
  - Independent personal expense tracker supporting INR (₹) formatting, category donut charts, monthly trends, and statistical anomaly detection.
- **Feature Explainability & Insights:**
  - Gini importance charts for tree models and theoretical breakdowns of PCA components.

---

## 🛠️ 3. Technology Stack

- **Core Language:** Python 3.12
- **Web Application & UI:** Streamlit (Multi-page architecture, responsive layout, custom fintech styling)
- **Data Manipulation:** Pandas, NumPy
- **Machine Learning:** Scikit-learn (`ColumnTransformer`, `RobustScaler`, `LogisticRegression`, `DecisionTreeClassifier`, `RandomForestClassifier`)
- **Imbalance Handling:** Imbalanced-learn (`SMOTE`)
- **Visualizations:** Plotly Graph Objects & Plotly Express
- **Model Serialization:** Joblib
- **Testing Suite:** Pytest

---

## 📂 4. Project Folder Structure

```
finguard-ai/
├── app.py                     # Main Streamlit Application (Executive Dashboard)
├── config.py                  # Central configuration (schemas, paths, thresholds, colors)
├── requirements.txt           # Python dependency specifications
├── pytest.ini                 # Pytest configuration
├── .gitignore                 # Version control exclusions
├── data/
│   ├── README.md              # Kaggle dataset download guide
│   ├── sample_transactions.csv# Synthetic demo transactions for instant testing
│   └── sample_spending.csv    # Sample personal spending dataset
├── models/
│   ├── README.md              # Documentation on saved model artifacts
│   ├── preprocessor.joblib    # Fitted RobustScaler ColumnTransformer
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── random_forest.joblib
│   └── metadata.json          # Actual evaluation metrics on untouched test sets
├── src/
│   ├── __init__.py
│   ├── data_loader.py         # CSV loading, schema validation, synthetic generator
│   ├── preprocessing.py      # Stratified splitting, RobustScaler, optional SMOTE
│   ├── train.py               # Multi-model training and evaluation pipeline
│   ├── predict.py             # Single & batch inference engine with caching
│   ├── evaluation.py          # Metric calculations, confusion matrices, curves
│   ├── spending_analytics.py  # Personal expense calculations and anomaly detection
│   ├── visualisations.py      # Cohesive dark Plotly charts and heatmaps
│   └── ui_components.py       # Reusable CSS and sidebar telemetry components
├── pages/
│   ├── 1_Fraud_Detection.py      # Single & batch inference with threshold tuning
│   ├── 2_Transaction_Explorer.py  # Searchable transaction explorer with filters
│   ├── 3_Model_Lab.py             # Multi-model comparison, confusion matrix, retrain suite
│   ├── 4_Spending_Analytics.py   # Spending analytics and outlier detection
│   ├── 5_Model_Insights.py       # Feature importances and PCA interpretability
│   └── 6_About_Project.py        # Academic documentation and viva defense guide
├── tests/
│   ├── test_data_validation.py    # Schema checks, missing columns, invalid labels
│   ├── test_preprocessing.py      # Splitting, scaling, and leakage prevention
│   ├── test_prediction.py         # Probability range and threshold logic
│   ├── test_evaluation.py         # Metrics calculations and tradeoff explanations
│   └── test_spending_analytics.py # Spending aggregations and anomaly detection
└── docs/
    ├── PROJECT_REPORT.md      # Full diploma semester academic project report
    ├── VIVA_QUESTIONS.md      # 30+ comprehensive viva defense questions & answers
    └── ARCHITECTURE.md        # System architecture and data flow diagrams
```

---

## 📥 5. Dataset Instructions (Kaggle Benchmark)

To train on the official European credit card fraud benchmark dataset:
1. Download `creditcard.csv` from [Kaggle MLG-ULB Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
2. Place the file inside the `data/` folder:
   ```
   data/creditcard.csv
   ```
3. The dataset contains 284,807 transactions with 492 fraud cases across 30 input features (`Time`, `V1`–`V28`, `Amount`) and target `Class`.
4. **Offline Demo Mode:** If the 150MB file has not yet been downloaded, FinGuard AI runs seamlessly out of the box using `data/sample_transactions.csv` or by generating deterministic synthetic demo batches inside the UI.

---

## 🚀 6. Installation & Execution Guide

### Step 1: Clone or Navigate to the Project
```bash
cd "c:/DIPLOMA PROJECTS/finguard-ai"
```

### Step 2: Create and Activate a Virtual Environment

**On Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**On macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run the Machine Learning Training Pipeline
You can trigger the genuine Scikit-learn training pipeline directly via terminal:
```bash
python -m src.train
```
*(Alternatively, you can retrain with a single click inside the Streamlit **Model Lab** page).*

### Step 5: Launch the Streamlit Web Application
```bash
streamlit run app.py
```
Open your browser and navigate to `http://localhost:8501`.

---

## 🧪 7. Automated Testing Suite

FinGuard AI includes 17 unit tests verifying schema validation, data leakage prevention, probability outputs, threshold classification, and spending aggregations.

Run the test suite with:
```bash
pytest -v
```

Expected output:
```text
tests/test_data_validation.py::test_valid_synthetic_dataset PASSED
tests/test_data_validation.py::test_missing_required_columns PASSED
tests/test_data_validation.py::test_single_class_target PASSED
tests/test_data_validation.py::test_invalid_target_values PASSED
tests/test_data_validation.py::test_non_numeric_features PASSED
tests/test_evaluation.py::test_evaluate_predictions_metrics PASSED
tests/test_evaluation.py::test_explain_metric_tradeoff PASSED
tests/test_prediction.py::test_inference_artifacts_load PASSED
tests/test_prediction.py::test_single_prediction_format_and_threshold PASSED
tests/test_prediction.py::test_batch_prediction PASSED
tests/test_preprocessing.py::test_separate_features_and_target PASSED
tests/test_preprocessing.py::test_stratified_train_test_split PASSED
tests/test_preprocessing.py::test_preprocessor_scaling PASSED
tests/test_spending_analytics.py::test_generate_and_standardize_spending PASSED
tests/test_spending_analytics.py::test_compute_spending_metrics PASSED
tests/test_spending_analytics.py::test_category_breakdown PASSED
tests/test_spending_analytics.py::test_detect_spending_anomalies PASSED

============================= 17 passed in 2.48s ==============================
```

---

## 👥 8. Student Team Contributions

| Student | Core Domain | Responsibilities |
| :--- | :--- | :--- |
| **Student A** | ML Pipeline & Data Engineering | Implemented dataset validation, stratified train/test splitting, RobustScaler pipeline, SMOTE, and training algorithms. |
| **Student B** | Frontend UI & Visualizations | Created Streamlit multi-page interface, custom fintech CSS theme, Plotly charts, threshold tuning sliders, and batch inference. |
| **Student C** | Spending Analytics, QA & Docs | Developed spending analytics module, wrote 17 Pytest unit tests, compiled viva defense documentation, and project reports. |

---

## 📜 9. License & Dataset Attribution
- **Dataset Attribution:** Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson, and Gianluca Bontempi. *Calibrating Probability with Undersampling for Unbalanced Classification.* In IEEE SSCI 2015.
- **Academic Use:** This project is prepared strictly for educational purposes as a diploma semester mini-project.
