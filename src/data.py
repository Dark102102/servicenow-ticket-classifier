"""Shared paths, constants, and data-loading helpers used by every notebook."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "raw" / "all_tickets_processed_improved_v3.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
FIGURES_DIR = ROOT / "figures"
RESULTS_DIR = ROOT / "results"

TEXT_COL = "Document"
LABEL_COL = "Topic_group"
SEED = 42

# Classes ordered by frequency (largest first); every chart and confusion matrix uses this order.
LABELS = [
    "Hardware",
    "HR Support",
    "Access",
    "Miscellaneous",
    "Storage",
    "Purchase",
    "Internal Project",
    "Administrative rights",
]


def load_split(name: str) -> pd.DataFrame:
    """Load a saved split: 'train', 'val' or 'test' (columns: text, label)."""
    return pd.read_csv(PROCESSED_DIR / f"{name}.csv")


def load_raw() -> pd.DataFrame:
    """Load the original Kaggle CSV and add a word-count column."""
    df = pd.read_csv(RAW_CSV)
    df["n_words"] = df[TEXT_COL].str.split().str.len()
    return df
