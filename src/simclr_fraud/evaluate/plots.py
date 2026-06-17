"""Evaluation plots for fraud classifier assessment."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def _closest_threshold_idx(thresholds: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(thresholds - target)))


def threshold_metrics_table(
    y_true: np.ndarray,
    y_score: np.ndarray,
    thresholds: list[float],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for target in thresholds:
        preds = y_score >= target
        tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        rows.append(
            {
                "threshold": target,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "tn": int(tn),
                "predicted_fraud_count": int(preds.sum()),
            }
        )
    return pd.DataFrame(rows)


def find_best_f1_threshold(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    f1_scores = np.zeros_like(thresholds, dtype=float)
    for i, t in enumerate(thresholds):
        p, r = precision[i], recall[i]
        f1_scores[i] = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    best_idx = int(np.argmax(f1_scores))
    return float(thresholds[best_idx]), float(f1_scores[best_idx])


def plot_roc_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    marker_thresholds: list[float],
    title: str = "ROC Curve",
) -> plt.Figure:
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_score)
    roc_auc = roc_auc_score(y_true, y_score)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, linewidth=2, label=f"ROC (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1, label="Random")

    for target in marker_thresholds:
        idx = _closest_threshold_idx(roc_thresholds, target)
        ax.scatter(
            fpr[idx],
            tpr[idx],
            s=80,
            zorder=5,
            label=f"θ={target}",
        )
        ax.annotate(
            f"{target}",
            (fpr[idx], tpr[idx]),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate / Recall")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_pr_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    marker_thresholds: list[float],
    title: str = "Precision-Recall Curve",
) -> plt.Figure:
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_score)
    pr_auc = average_precision_score(y_true, y_score)
    fraud_rate = float(y_true.mean())

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall[:-1], precision[:-1], linewidth=2, label=f"PR (AP = {pr_auc:.4f})")
    ax.axhline(fraud_rate, color="gray", linestyle="--", linewidth=1, label=f"Random ({fraud_rate:.4f})")

    for target in marker_thresholds:
        idx = _closest_threshold_idx(pr_thresholds, target)
        ax.scatter(recall[idx], precision[idx], s=80, zorder=5)
        ax.annotate(
            f"{target}",
            (recall[idx], precision[idx]),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_threshold_sweep(threshold_df: pd.DataFrame, title: str = "Metrics vs Threshold") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(threshold_df["threshold"], threshold_df["precision"], marker="o", label="Precision")
    ax.plot(threshold_df["threshold"], threshold_df["recall"], marker="o", label="Recall")
    ax.plot(threshold_df["threshold"], threshold_df["f1"], marker="o", label="F1")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_score_distribution(
    y_true: np.ndarray,
    y_score: np.ndarray,
    title: str = "Predicted Fraud Score Distribution",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(y_score[y_true == 0], bins=50, alpha=0.6, density=True, label="Non-fraud")
    ax.hist(y_score[y_true == 1], bins=50, alpha=0.6, density=True, label="Fraud")
    ax.set_xlabel("Predicted fraud probability")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    title: str | None = None,
) -> plt.Figure:
    preds = y_score >= threshold
    cm = confusion_matrix(y_true, preds, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred 0", "Pred 1"])
    ax.set_yticklabels(["True 0", "True 1"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title or f"Confusion Matrix (θ={threshold})")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    return fig
