"""
FinGuard AI - Configuration Module
Centralized configuration for dataset schemas, model parameters, paths, and UI settings.
"""

from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
DOCS_DIR = PROJECT_ROOT / "docs"

# Ensure runtime directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Standard Credit Card Dataset Schema (Kaggle MLG-ULB)
EXPECTED_FEATURES = [
    "Time",
    "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
    "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20",
    "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28",
    "Amount"
]
TARGET_COLUMN = "Class"
ALL_COLUMNS = EXPECTED_FEATURES + [TARGET_COLUMN]

# Dataset filenames
REAL_DATASET_NAME = "creditcard.csv"
REAL_DATASET_PATH = DATA_DIR / REAL_DATASET_NAME
SAMPLE_DATASET_PATH = DATA_DIR / "sample_transactions.csv"
SAMPLE_SPENDING_PATH = DATA_DIR / "sample_spending.csv"

# Preprocessing & Split Config
TEST_SIZE = 0.20
RANDOM_STATE = 42

# Default Decision Threshold for Fraud Probability
DEFAULT_THRESHOLD = 0.50

# UI Fintech Theme Colors
THEME = {
    "primary": "#1E3A8A",      # Deep Navy / Royal Blue
    "accent": "#3B82F6",       # Bright Blue Accent
    "success": "#10B981",      # Emerald Green (Legitimate)
    "warning": "#F59E0B",      # Amber (Suspicious / Moderate Risk)
    "danger": "#EF4444",       # Crimson Red (Fraudulent)
    "background": "#0F172A",   # Dark Slate Background
    "surface": "#1E293B",      # Dark Card Surface
    "surface_light": "#F8FAFC",# Clean Neutral Light
    "text_dark": "#0F172A",
    "text_light": "#F8FAFC",
    "muted": "#94A3B8"
}
