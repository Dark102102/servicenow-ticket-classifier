"""Evaluation helpers shared by every model, so all results are computed and plotted the same way."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)

from src.data import LABELS, RESULTS_DIR
from src.plotting import SURFACE, SERIES, TEXT, save

CM_CMAP = LinearSegmentedColormap.from_list("seq_blue", [SURFACE, "#a9c8ee", SERIES, "#123e75"])


def summary_metrics(y_true, y_pred) -> dict:
    """Headline metrics for the model-comparison table."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def per_class_report(y_true, y_pred) -> pd.DataFrame:
    rep = classification_report(y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0)
    return pd.DataFrame({l: rep[l] for l in LABELS}).T.rename(columns={"f1-score": "f1"})


def record_metrics(model_name: str, split: str, y_true, y_pred, **extra) -> pd.DataFrame:
    """Add/replace one row in results/metrics.csv and return the whole table."""
    path = RESULTS_DIR / "metrics.csv"
    row = {"model": model_name, "split": split, **summary_metrics(y_true, y_pred), **extra}
    table = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if not table.empty:
        table = table[~((table.model == model_name) & (table.split == split))]
    table = pd.concat([table, pd.DataFrame([row])], ignore_index=True)
    table.to_csv(path, index=False)
    return table


def save_predictions(model_name: str, texts, y_true, proba: np.ndarray) -> pd.DataFrame:
    """Save test-set predictions + class probabilities (used later for threshold and error analysis)."""
    preds = pd.DataFrame(proba, columns=[f"p_{l}" for l in LABELS])
    preds.insert(0, "text", list(texts))
    preds.insert(1, "label", list(y_true))
    preds.insert(2, "pred", np.array(LABELS)[proba.argmax(1)])
    preds.insert(3, "confidence", proba.max(1))
    preds.to_csv(RESULTS_DIR / f"preds_{model_name}.csv", index=False)
    return preds


def plot_confusion_matrix(y_true, y_pred, title: str, fig_name: str):
    """Row-normalised confusion matrix: each row shows where that group's tickets were routed (%)."""
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    pct = cm / cm.sum(axis=1, keepdims=True) * 100
    fig, ax = plt.subplots(figsize=(8.5, 7))
    ax.imshow(pct, cmap=CM_CMAP, vmin=0, vmax=100)
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            v = pct[i, j]
            if v >= 0.5:
                ax.text(j, i, f"{v:.0f}%\n({cm[i, j]})", ha="center", va="center", fontsize=8,
                        color="white" if v > 55 else TEXT)
    ax.set_xticks(range(len(LABELS)), LABELS, rotation=35, ha="right")
    ax.set_yticks(range(len(LABELS)), LABELS)
    ax.set_xlabel("Predicted support group")
    ax.set_ylabel("Actual support group")
    ax.grid(False)
    ax.set_title(title)
    save(fig, fig_name)
    return fig, cm
