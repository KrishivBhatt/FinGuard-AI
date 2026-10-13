"""
FinGuard AI - Personal Spending Analytics Module
Independent analytical module for personal expense tracking, category breakdowns,
monthly trends, and experimental anomaly detection.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

DEFAULT_CURRENCY = "₹"

def validate_and_standardize_spending_df(
    df: pd.DataFrame,
    date_col: str = "Date",
    amount_col: str = "Amount",
    category_col: str = "Category",
    desc_col: Optional[str] = "Description"
) -> Tuple[Optional[pd.DataFrame], List[str]]:
    """
    Validates and standardizes a user-uploaded spending CSV.
    Maps arbitrary column names to standard fields.
    """
    errors = []
    if df is None or df.empty:
        errors.append("Spending dataset is empty.")
        return None, errors

    for required_field, actual_col in [("Date", date_col), ("Amount", amount_col)]:
        if actual_col not in df.columns:
            errors.append(f"Required column '{required_field}' (mapped to '{actual_col}') not found in dataset.")

    if errors:
        return None, errors

    clean_df = pd.DataFrame()
    # Parse dates
    try:
        clean_df["Date"] = pd.to_datetime(df[date_col], errors="coerce")
        if clean_df["Date"].isnull().all():
            errors.append(f"Could not parse dates from column '{date_col}'. Please verify date format (e.g., YYYY-MM-DD).")
            return None, errors
    except Exception as e:
        errors.append(f"Error parsing dates: {str(e)}")
        return None, errors

    # Parse amounts
    try:
        clean_df["Amount"] = pd.to_numeric(df[amount_col], errors="coerce").fillna(0.0)
    except Exception as e:
        errors.append(f"Error parsing amounts: {str(e)}")
        return None, errors

    # Category
    if category_col in df.columns:
        clean_df["Category"] = df[category_col].astype(str).str.strip().replace("", "Miscellaneous")
    else:
        clean_df["Category"] = "Uncategorized"

    # Description
    if desc_col and desc_col in df.columns:
        clean_df["Description"] = df[desc_col].astype(str)
    else:
        clean_df["Description"] = "Transaction"

    # Drop invalid rows and sort chronologically
    clean_df = clean_df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    return clean_df, []


def compute_spending_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes key performance metrics for spending.
    """
    if df is None or df.empty:
        return {
            "total_spending": 0.0,
            "transaction_count": 0,
            "avg_transaction": 0.0,
            "median_transaction": 0.0,
            "max_transaction": 0.0,
            "min_transaction": 0.0,
            "date_range": ("N/A", "N/A"),
            "top_category": "N/A"
        }

    total = float(df["Amount"].sum())
    count = len(df)
    avg = float(df["Amount"].mean())
    median = float(df["Amount"].median())
    max_amt = float(df["Amount"].max())
    min_amt = float(df["Amount"].min())
    min_date = df["Date"].min().strftime("%d %b %Y")
    max_date = df["Date"].max().strftime("%d %b %Y")

    category_sums = df.groupby("Category")["Amount"].sum()
    top_cat = str(category_sums.idxmax()) if not category_sums.empty else "N/A"

    return {
        "total_spending": round(total, 2),
        "transaction_count": count,
        "avg_transaction": round(avg, 2),
        "median_transaction": round(median, 2),
        "max_transaction": round(max_amt, 2),
        "min_transaction": round(min_amt, 2),
        "date_range": (min_date, max_date),
        "top_category": top_cat
    }


def get_category_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns breakdown of spending by category with totals and percentages.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["Category", "Total_Amount", "Percentage", "Count"])

    total_sum = df["Amount"].sum()
    grouped = df.groupby("Category").agg(
        Total_Amount=("Amount", "sum"),
        Count=("Amount", "count")
    ).reset_index()

    grouped["Percentage"] = (grouped["Total_Amount"] / total_sum * 100.0).round(2) if total_sum > 0 else 0.0
    grouped = grouped.sort_values("Total_Amount", ascending=False).reset_index(drop=True)
    return grouped


def get_monthly_spending_trend(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates spending into monthly periods.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["Month_Year", "Total_Amount", "Transaction_Count"])

    temp = df.copy()
    temp["Month_Year"] = temp["Date"].dt.to_period("M").astype(str)
    monthly = temp.groupby("Month_Year").agg(
        Total_Amount=("Amount", "sum"),
        Transaction_Count=("Amount", "count")
    ).reset_index()

    monthly["Total_Amount"] = monthly["Total_Amount"].round(2)
    return monthly


def detect_spending_anomalies(df: pd.DataFrame, threshold_std: float = 2.5) -> pd.DataFrame:
    """
    Experimental spending anomaly detector based on statistical deviation (Z-score).
    Labelled as EXPERIMENTAL: Separate from the supervised PCA fraud classifier.
    """
    if df is None or len(df) < 5:
        return pd.DataFrame()

    temp = df.copy()
    mean = temp["Amount"].mean()
    std = temp["Amount"].std()

    if std == 0:
        return pd.DataFrame()

    temp["Z_Score"] = ((temp["Amount"] - mean) / std).abs()
    anomalies = temp[temp["Z_Score"] >= threshold_std].copy()
    anomalies["Status"] = "Unusually High Expense"
    return anomalies.sort_values("Amount", ascending=False)


def generate_sample_spending_data(n_records: int = 150) -> pd.DataFrame:
    """
    Generates a realistic sample personal spending history (Indian retail context).
    """
    rng = np.random.RandomState(42)
    start_date = datetime.now() - timedelta(days=120)

    categories_and_ranges = [
        ("Groceries & Supermarket", (300, 3500), ["BigBasket", "Blinkit", "DMart", "Local Kirana", "Nature's Basket"]),
        ("Dining & Food Delivery", (150, 1800), ["Zomato", "Swiggy", "Barbeque Nation", "Haldirams", "Starbucks"]),
        ("Utilities & Bills", (500, 4500), ["Electricity Bill", "Broadband Bill", "Mobile Recharge", "Piped Gas", "Water Bill"]),
        ("Shopping & Electronics", (600, 12000), ["Amazon.in", "Flipkart", "Myntra", "Croma", "Zara"]),
        ("Transportation & Fuel", (80, 2200), ["Uber", "Ola", "Metro Card", "HP Petrol Pump", "Fastag"]),
        ("Healthcare & Pharmacy", (200, 3000), ["Apollo Pharmacy", "1mg", "Dental Clinic", "Diagnostic Lab"]),
        ("Entertainment & OTT", (199, 1499), ["Netflix", "BookMyShow", "Spotify", "Hotstar", "PlayStation Store"]),
        ("Education & Courses", (499, 5000), ["Udemy", "Coursera", "Technical Books", "Exam Fee"])
    ]

    records = []
    for _ in range(n_records):
        cat_tuple = categories_and_ranges[rng.randint(len(categories_and_ranges))]
        cat_name, (low, high), vendors = cat_tuple

        # Exponential-skewed realistic amounts
        amt = round(float(rng.uniform(low, high)), 2)
        days_offset = int(rng.uniform(0, 120))
        txn_date = start_date + timedelta(days=days_offset)
        vendor = vendors[rng.randint(len(vendors))]

        records.append({
            "Date": txn_date.strftime("%Y-%m-%d"),
            "Amount": amt,
            "Category": cat_name,
            "Description": f"{vendor} Payment"
        })

    # Add 2 large unusual expenses for anomaly testing
    records.append({
        "Date": (datetime.now() - timedelta(days=15)).strftime("%Y-%m-%d"),
        "Amount": 48500.00,
        "Category": "Shopping & Electronics",
        "Description": "Apple Store / Laptop Purchase"
    })
    records.append({
        "Date": (datetime.now() - timedelta(days=45)).strftime("%Y-%m-%d"),
        "Amount": 28000.00,
        "Category": "Healthcare & Pharmacy",
        "Description": "Hospital Medical Procedure"
    })

    df = pd.DataFrame(records)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df
