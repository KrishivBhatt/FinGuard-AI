# ACADEMIC SEMESTER MINI-PROJECT REPORT

---

## **FinGuard AI: Machine Learning-Based Credit Card Fraud Detection and Spending Analytics System**

**Submitted in partial fulfillment of the requirements for the Diploma in Computer Engineering / Computer Science**

**Academic Year:** 2025–2026  
**Department:** Computer Engineering / Information Technology  
**Project Guide / Supervisor:** [Supervisor Name, Title]  
**Project Team Members:**
1. **[Student A Name]** — Roll No: [XXXXX] (Data Engineering & ML Algorithms)
2. **[Student B Name]** — Roll No: [XXXXX] (Frontend & Interactive Visualizations)
3. **[Student C Name]** — Roll No: [XXXXX] (Analytics, Quality Assurance & Documentation)

---

### **TABLE OF CONTENTS**
1. [Abstract](#1-abstract)
2. [Introduction](#2-introduction)
3. [Problem Definition & Motivation](#3-problem-definition--motivation)
4. [Existing System & Limitations](#4-existing-system--limitations)
5. [Proposed System Architecture](#5-proposed-system-architecture)
6. [Hardware & Software Requirements](#6-hardware--software-requirements)
7. [Dataset Description & Preprocessing](#7-dataset-description--preprocessing)
8. [Machine Learning Algorithms](#8-machine-learning-algorithms)
9. [Performance Evaluation & Metrics](#9-performance-evaluation--metrics)
10. [Application Screenshots Guide](#10-application-screenshots-guide)
11. [Limitations & Ethical Considerations](#11-limitations--ethical-considerations)
12. [Future Scope](#12-future-scope)
13. [Conclusion](#13-conclusion)
14. [References](#14-references)

---

### **1. ABSTRACT**
In digital banking and electronic commerce, credit card fraud represents a critical threat resulting in billions of dollars in losses annually. The core technical hurdle in detecting financial fraud lies in the extreme class imbalance of transaction datasets, where legitimate transactions constitute greater than 99.8% of volume. Conventional evaluation metrics, such as basic classification accuracy, produce deceptive results, masking an algorithm's failure to identify rare fraudulent records.

This project presents **FinGuard AI**, an end-to-end fintech machine learning application developed in Python and Streamlit. The system establishes a leakage-free machine learning pipeline evaluating three supervised algorithms: **Logistic Regression**, **Decision Tree Classifier**, and **Random Forest Classifier**. By leveraging stratified train/test partitions, Robust feature scaling, and Precision-Recall Area Under Curve (PR-AUC) metrics, FinGuard AI ensures mathematically rigorous fraud detection. The platform includes an interactive single and batch inference engine with dynamic decision threshold tuning, an exploratory transaction browser, and an independent personal spending analytics module.

---

### **2. INTRODUCTION**
Financial technologies (Fintech) rely heavily on automated risk management engines to verify card transactions in milliseconds. Modern machine learning techniques enable financial institutions to move away from static, rule-based heuristics toward adaptive, pattern-based classification models capable of identifying non-linear spending deviations.

FinGuard AI addresses the complete lifecycle of financial fraud modeling, from dataset ingestion and schema validation to feature scaling, stratified evaluation, model persistence, and interactive user presentation.

---

### **3. PROBLEM DEFINITION & MOTIVATION**
#### 3.1 Problem Definition
Given a financial transaction record consisting of time of execution, monetary amount, and statistical principal component features ($V1$ through $V28$), predict whether the transaction is **Legitimate ($Class = 0$)** or **Fraudulent ($Class = 1$)**, outputting a well-calibrated fraud probability and allowing risk officer intervention via custom thresholding.

#### 3.2 Motivation
- **Financial Loss Mitigation:** Preventing unauthorized card charges before batch clearing.
- **Minimizing Cardholder Friction:** Excessive false alerts cause genuine transactions to be rejected, causing customer churn.
- **Academic Learning:** Demonstrating mastery of real-world machine learning challenges, including class imbalance, data leakage prevention, and model persistence.

---

### **4. EXISTING SYSTEM & LIMITATIONS**
Traditional banking fraud detection relies primarily on manual rule-based thresholds (e.g., flagging transactions exceeding $5,000 or originating from blacklisted IP ranges).

| Aspect | Existing Rule-Based Systems | Proposed FinGuard AI System |
| :--- | :--- | :--- |
| **Adaptability** | Rigid, requires manual rule updates by risk analysts | Adaptive, learns multi-dimensional feature boundaries |
| **Non-linear Patterns** | Fails to detect complex feature interactions | Trees and Ensembles model non-linear interactions |
| **Imbalance Handling** | Prone to overwhelming alert volume or high false alarms | Imbalance-aware algorithms using PR-AUC and cost weighting |
| **Interactivity** | Static batch reports | Real-time threshold tuning and live CSV batch scoring |

---

### **5. PROPOSED SYSTEM ARCHITECTURE**
FinGuard AI is structured into decoupled, modular layers:
1. **Data Ingestion & Validation Layer (`src/data_loader.py`):** Schema inspection, null checks, class distribution validation.
2. **Preprocessing Layer (`src/preprocessing.py`):** Stratified 80/20 splitting, Scikit-learn `RobustScaler` ColumnTransformer fitted strictly on training data.
3. **Training & Evaluation Layer (`src/train.py`, `src/evaluation.py`):** Model fitting, confusion matrix computation, PR-AUC and ROC-AUC curve generation, model serialization with `joblib`.
4. **Inference Engine (`src/predict.py`):** Probability calculation, threshold-based classification, single and batch scoring with memory caching.
5. **Presentation Layer (`app.py`, `pages/*.py`):** Multi-page Streamlit interface with dark fintech theme, Plotly interactive graphics, and personal spending analytics.

---

### **6. HARDWARE & SOFTWARE REQUIREMENTS**
#### 6.1 Hardware Requirements
- **Processor:** Dual-Core Intel/AMD CPU (Quad-core or higher recommended for Random Forest training).
- **RAM:** Minimum 4 GB (8 GB recommended for full 284k Kaggle dataset).
- **Disk Storage:** Minimum 1 GB free space.

#### 6.2 Software Requirements
- **Operating System:** Windows 10/11, macOS, or Linux.
- **Programming Language:** Python 3.10–3.12.
- **Libraries:** Streamlit, Pandas, NumPy, Scikit-learn, Imbalanced-learn, Plotly, Joblib, Pytest.

---

### **7. DATASET DESCRIPTION & PREPROCESSING**
#### 7.1 Kaggle Benchmark Dataset
The system is built for the **MLG-ULB Credit Card Fraud Detection** benchmark dataset:
- **Total Transactions:** 284,807 records.
- **Fraudulent Transactions:** 492 records (~0.172% fraud prevalence).
- **Features ($V1$ to $V28$):** Transformed numerical variables resulting from Principal Component Analysis (PCA) to preserve cardholder confidentiality.
- **Un-transformed Features:**
  - `Time`: Elapsed seconds since the first recorded transaction in the dataset.
  - `Amount`: Total monetary value of the transaction.
  - `Class`: Ground truth binary label ($0 = \text{Legitimate}, 1 = \text{Fraudulent}$).

#### 7.2 Data Leakage Prevention
To guarantee authentic generalization:
- The dataset is split into training (80%) and testing (20%) using **stratified sampling**.
- The `ColumnTransformer` (fitting `RobustScaler` on `Time` and `Amount`) is **fitted exclusively on the training partition**. The test partition is transformed using the fitted training parameters without re-fitting.
- Oversampling techniques (such as SMOTE) are restricted strictly to the training fold.

---

### **8. MACHINE LEARNING ALGORITHMS**
1. **Logistic Regression (Baseline):**
   - Applies sigmoid activation over linear feature combinations.
   - Utilizes `class_weight='balanced'` to offset the penalty proportional to class imbalance.
2. **Decision Tree Classifier (Nonlinear & Interpretable):**
   - Partitions feature space by minimizing Gini impurity.
   - Constrained with `max_depth=6` to prevent memorization and overfitting.
3. **Random Forest Classifier (Ensemble Benchmark):**
   - Combines 100 decorrelated decision trees using bootstrap aggregating (bagging).
   - Generates robust probability estimates and reduces variance across noisy PCA coordinates.

---

### **9. PERFORMANCE EVALUATION & METRICS**
#### 9.1 Evaluation Philosophy
Because legitimate transactions make up over 99.8% of records, **Accuracy is an invalid primary evaluation metric**. The primary benchmarks are:
- **Recall (Sensitivity):** $\frac{TP}{TP + FN}$ — Ability to capture fraudulent instances.
- **Precision:** $\frac{TP}{TP + FP}$ — Accuracy of positive fraud alerts.
- **F1-Score:** Harmonic mean of precision and recall.
- **PR-AUC (Average Precision):** Area under the Precision-Recall curve, directly assessing positive class trade-offs across all classification thresholds.

#### 9.2 Measured Experimental Results (Genuine Kaggle Benchmark Run)
Evaluated on untouched test partition (Stratified 20% holdout = 56,962 transactions, 98 fraud cases):

| Supervised Model | Test Accuracy | Test Recall (Sensitivity) | Test Precision | F1-Score | PR-AUC (Avg Precision) | ROC-AUC | Training Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** (Balanced) | 97.55% | **91.84%** (90/98 fraud caught) | 6.09% | 0.1142 | 0.7175 | 0.9734 | 14.8s |
| **Decision Tree** (Max Depth 6) | 97.63% | 86.73% (85/98 fraud caught) | 5.97% | 0.1118 | 0.4643 | 0.9238 | 9.2s |
| **Random Forest** (100 Trees) | **99.92%** | 84.69% (83/98 fraud caught) | **73.45%** (83/113 flagged) | **0.7867** | **0.8358** | **0.9572** | 35.1s |

**Empirical Analysis:**
- **Logistic Regression** achieved the highest raw Recall (91.84%), capturing almost all fraud attempts, but generated 1,388 false positives due to linear boundary approximation under severe class weighting.
- **Decision Tree** showed lower PR-AUC (0.4643), as single orthogonal axis splits struggle with subtle correlations across high-dimensional PCA components.
- **Random Forest** proved to be the **optimal production model**, delivering an exceptional **PR-AUC of 0.8358** and **F1-Score of 0.7867**. It detected 84.69% of all fraud cases while generating only 30 false alarms across the entire 56,864 legitimate test transactions (73.45% precision), achieving the best operational balance for financial institutions.

---

### **10. APPLICATION SCREENSHOTS GUIDE**
To include actual application screenshots in your submitted semester report:
1. **Executive Dashboard (`app.py`):** Capture the system status panel, transaction telemetry KPI cards, and transaction distribution plot.
2. **Fraud Detection Engine (`pages/1_Fraud_Detection.py`):** Capture single transaction scoring with the risk banner and threshold slider set to 0.50.
3. **Batch Inference Interface:** Capture the scored CSV preview and export button.
4. **Transaction Explorer (`pages/2_Transaction_Explorer.py`):** Capture filtered rows with model agreement percentage and pagination.
5. **Model Lab Benchmark (`pages/3_Model_Lab.py`):** Capture the multi-metric comparative bar chart, confusion matrix heatmap, and PR curve.
6. **Spending Analytics (`pages/4_Spending_Analytics.py`):** Capture the category donut chart, monthly trajectory line chart, and outlier spending list.
7. **Model Insights (`pages/5_Model_Insights.py`):** Capture the Gini feature importance chart illustrating top PCA contributors.

---

### **11. LIMITATIONS & ETHICAL CONSIDERATIONS**
- **Offline Batch Architecture:** In this academic implementation, transactions are evaluated from CSV files rather than an event-streaming Kafka cluster.
- **PCA Anonymity:** Raw merchant details, cardholder demographics, and merchant categories cannot be reverse-engineered from $V1-V28$.
- **Ethical Decisioning:** Algorithmic outputs are decision-support probability scores, requiring human compliance review before blocking accounts.

---

### **12. FUTURE SCOPE**
- **Explainable AI (XAI) with SHAP:** Implement KernelSHAP for single-transaction local waterfall explanations.
- **Streaming Pipeline Integration:** Connect model inference endpoints to Apache Kafka and FastAPI microservices.
- **Concept Drift Detection:** Integrate statistical drift detectors (e.g., Kolmogorov-Smirnov tests) to flag shifts in spending behavior over time.

---

### **13. CONCLUSION**
The FinGuard AI system demonstrates an end-to-end implementation of an applied machine learning fintech solution for credit card fraud detection and spending analytics. By enforcing strict leakage-free preprocessing, comparing three genuine classification algorithms, and prioritizing PR-AUC over basic accuracy, the project adheres to professional data science standards while remaining accessible for diploma curriculum viva defense.

---

### **14. REFERENCES**
1. Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson, and Gianluca Bontempi. *Calibrating Probability with Undersampling for Unbalanced Classification.* IEEE SSCI 2015.
2. Pedregosa et al. *Scikit-learn: Machine Learning in Python.* JMLR 12, pp. 2825-2830, 2011.
3. Streamlit Documentation. *Streamlit Framework for Data Science Applications.* 2024.
4. Fawcett, Tom. *An introduction to ROC analysis.* Pattern Recognition Letters 27.8 (2006): 861-874.
