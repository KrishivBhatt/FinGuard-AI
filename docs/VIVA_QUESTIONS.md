# FinGuard AI — Viva Defense Questions & Answers

This guide is designed for **Third-Year Computer Science Diploma students** preparing for their semester project viva and external examination.

---

### **Q1: What is Machine Learning?**
**Answer:** Machine Learning (ML) is a branch of Artificial Intelligence where computer algorithms learn patterns and rules directly from historical data, allowing systems to make predictions or decisions on unseen data without being explicitly programmed with fixed static rules.

---

### **Q2: What is Supervised Learning?**
**Answer:** Supervised Learning is a category of machine learning where the training dataset includes both input features ($X$) and corresponding known ground-truth labels ($y$). The model learns a mathematical mapping function $f(X) \\to y$. In FinGuard AI, the model learns to map transaction features to whether the transaction was Legitimate (0) or Fraud (1).

---

### **Q3: What is Classification?**
**Answer:** Classification is a supervised learning task where the output variable is categorical or discrete. Since our target is binary ($0$ or $1$), FinGuard AI performs **Binary Classification**.

---

### **Q4: What is Logistic Regression?**
**Answer:** Logistic Regression is a foundational linear classification algorithm. It computes a weighted linear combination of input features and passes the result through the sigmoid function $\\sigma(z) = \\frac{1}{1 + e^{-z}}$, producing a probability score between $0.0$ and $1.0$.

---

### **Q5: What is a Decision Tree Classifier?**
**Answer:** A Decision Tree is a non-linear classification algorithm that partitions data into subsets by asking a sequence of if-else feature questions. Splits are chosen to maximize the purity of the resulting child nodes, measured using criteria like **Gini Impurity** or **Information Gain**.

---

### **Q6: What is a Random Forest Classifier?**
**Answer:** Random Forest is an **ensemble learning method** that trains multiple independent decision trees (e.g., 100 trees) on random subsets of the data (bootstrap aggregating or bagging) and random subsets of features. Final predictions are aggregated by majority voting or averaging predicted probabilities, which significantly reduces variance and overfitting.

---

### **Q7: What is Overfitting?**
**Answer:** Overfitting occurs when a machine learning model learns the training data too closely, memorizing random noise and outliers rather than general patterns. An overfitted model performs exceptionally well on training data but poorly on unseen test data. We control overfitting in FinGuard AI by constraining tree depth (`max_depth=6`) and evaluating on untouched test sets.

---

### **Q8: What is Underfitting?**
**Answer:** Underfitting occurs when a model is too simplistic to capture the underlying relationships in the data. An underfitted model exhibits high bias and performs poorly on both training and testing datasets.

---

### **Q9: Why is this dataset considered "Imbalanced"?**
**Answer:** In the Kaggle credit card dataset, out of 284,807 transactions, only 492 are fraudulent (~0.172%). The ratio between legitimate and fraudulent transactions is approximately **578 to 1**. This severe disparity represents an extreme class imbalance.

---

### **Q10: What is SMOTE?**
**Answer:** SMOTE stands for **Synthetic Minority Over-sampling Technique**. Rather than duplicating existing minority records, SMOTE creates synthetic new minority instances along the line segments connecting existing nearest neighbors in feature space. In FinGuard AI, SMOTE is applied strictly to training partitions to prevent data leakage.

---

### **Q11: What is Data Leakage and how did you prevent it?**
**Answer:** Data Leakage occurs when information from outside the training partition (such as test set statistics) inadvertently influences model training, producing artificially optimistic performance that fails in production. In FinGuard AI, we prevented leakage by:
1. Performing an 80/20 Stratified Split *before* scaling.
2. Fitting `RobustScaler` *strictly on the training set*.
3. Transforming the test set using only the fitted training parameters.

---

### **Q12: Why is Accuracy misleading for credit card fraud detection?**
**Answer:** This is known as the **Accuracy Paradox**. Because 99.8% of transactions are legitimate, a naive model that predicts every transaction is legitimate would achieve **99.8% accuracy**, yet fail to detect a single fraudulent transaction. Therefore, accuracy alone is clinically useless for fraud systems.

---

### **Q13: What are Precision and Recall?**
**Answer:**
- **Recall (Detection Rate):** $\\frac{TP}{TP + FN}$ — Out of all actual fraudulent transactions, what percentage did the model catch? High recall is essential to minimize financial losses.
- **Precision:** $\\frac{TP}{TP + FP}$ — Out of all transactions flagged as fraudulent, what percentage was genuinely fraudulent? High precision prevents cardholder frustration from false alarms.

---

### **Q14: What is an F1-Score?**
**Answer:** The F1-Score is the harmonic mean of Precision and Recall:
$$F1 = 2 \\times \\frac{\\text{Precision} \\times \\text{Recall}}{\\text{Precision} + \\text{Recall}}$$
It provides a single balanced metric when seeking an equilibrium between catching fraud and minimizing false alerts.

---

### **Q15: What is a Confusion Matrix?**
**Answer:** A Confusion Matrix is a 2x2 table summarizing classification outcomes:
- **True Negatives (TN):** Legitimate transactions correctly predicted as legitimate.
- **False Positives (FP):** Legitimate transactions incorrectly flagged as fraud (Type I error).
- **False Negatives (FN):** Fraudulent transactions missed by the model (Type II error).
- **True Positives (TP):** Fraudulent transactions correctly caught.

---

### **Q16: What is ROC-AUC?**
**Answer:** ROC-AUC measures the Area Under the Receiver Operating Characteristic curve, plotting True Positive Rate against False Positive Rate across all decision thresholds. A random guess has an AUC of 0.5; a perfect classifier has an AUC of 1.0.

---

### **Q17: What is PR-AUC and why is it preferred over ROC-AUC here?**
**Answer:** PR-AUC is the Area Under the **Precision-Recall Curve**. In highly imbalanced datasets where True Negatives vastly outnumber True Positives, the False Positive Rate ($FP / (FP + TN)$) remains tiny even with many false positives, making ROC-AUC look deceptively high (e.g. 0.98). PR-AUC focuses directly on the minority class without being distorted by large TN counts.

---

### **Q18: What is Feature Scaling and why did you use RobustScaler?**
**Answer:** Feature scaling brings features onto a comparable numerical scale so variables with large magnitudes (like `Time` or `Amount`) do not artificially dominate distance-based or gradient calculations. We used `RobustScaler` because financial amounts contain extreme outliers; `RobustScaler` scales using the median and Interquartile Range (IQR), making it resilient to extreme values.

---

### **Q19: Why is a Train/Test split necessary?**
**Answer:** To simulate real-world production performance on unseen data. Evaluating a model on the same data it was trained on only measures memorization, not generalization.

---

### **Q20: Why are features V1 through V28 anonymized?**
**Answer:** Financial institutions are legally bound by privacy regulations (PCI-DSS, GDPR). The dataset providers applied **Principal Component Analysis (PCA)** to transform sensitive features (cardholder identity, merchant names, location) into anonymized orthogonal mathematical coordinates.

---

### **Q21: How is a fraud probability produced?**
**Answer:** In Scikit-learn, models expose a `.predict_proba()` method. Logistic Regression calculates it directly via the sigmoid function; Random Forest calculates it as the fraction of decision trees in the ensemble that voted for Class 1.

---

### **Q22: How does the Decision Threshold affect False Positives?**
**Answer:** The decision threshold is the probability cutoff above which a transaction is flagged as fraud (default = 0.50).
- Lowering the threshold (e.g., to 0.20) increases Recall (catches more fraud) but increases False Positives (more genuine cards flagged).
- Raising the threshold (e.g., to 0.80) reduces False Positives (fewer false alerts) but increases False Negatives (more fraud slips through).

---

### **Q23: Why is Random Forest generally preferred for tabular fraud data?**
**Answer:** Random Forest handles complex, non-linear feature interactions, is naturally robust to outliers, and combines predictions across 100 trees to achieve superior Precision-Recall AUC compared to single linear models.

---

### **Q24: What are the main limitations of this project?**
**Answer:**
1. It operates on historical offline batch CSV files rather than real-time streaming data pipelines (like Apache Kafka).
2. The PCA features cannot reveal semantic merchant or geographical details.
3. It does not account for continuous concept drift where fraud patterns evolve over months.

---

### **Q25: How could the system be improved in the future?**
**Answer:**
1. Integrating Explainable AI (SHAP / LIME) for local transaction explanation waterfalls.
2. Deploying model endpoints as microservices via FastAPI and Docker.
3. Adding concept drift monitoring using statistical drift detection libraries (e.g. Evidently AI).

---

### **Q26: What role does Joblib play in your architecture?**
**Answer:** `joblib` is used for efficient serialization and persistence of Python objects and NumPy arrays. It allows trained Scikit-learn models and preprocessing pipelines to be saved to disk and loaded instantly for inference without retraining on every page load.

---

### **Q27: How does your Spending Analytics module differ from the Fraud Detection module?**
**Answer:** The Fraud Detection module uses supervised machine learning on PCA transaction coordinates. The Spending Analytics module is an independent analytical tool that accepts user bank statements with `Date`, `Amount`, and `Category`, computing monthly trends, category budgets, and experimental Z-score outlier expenses.
