# FinGuard AI - Dataset Directory

This directory stores transaction datasets for FinGuard AI.

## 1. Real Kaggle Dataset (Recommended for Full Evaluation)
To use the official Kaggle Credit Card Fraud Detection benchmark dataset:
1. Download `creditcard.csv` from [Kaggle MLG-ULB Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
2. Save the file directly in this folder as:
   ```
   data/creditcard.csv
   ```
3. Dataset Attributes:
   - **Total Records:** 284,807 transactions.
   - **Fraudulent Records:** 492 transactions (~0.172%).
   - **Features:** `Time`, `V1` to `V28` (anonymized PCA components), `Amount`, and target `Class` (0 = Legitimate, 1 = Fraudulent).

## 2. Sample Datasets (Included for Quick Demonstration)
- `sample_transactions.csv`: A formatted sample containing both legitimate and fraudulent transactions matching the exact PCA feature schema. Used for rapid UI inspection and offline testing.
- `sample_spending.csv`: A personal spending history containing `Date`, `Amount`, `Category`, and `Description` for the Spending Analytics module.

## 3. Synthetic Demo Mode
If `creditcard.csv` is not yet placed, FinGuard AI features an automated **Synthetic Demo Generator** accessible directly within the UI, allowing full pipeline training, metric evaluation, and threshold testing without waiting for large downloads.
