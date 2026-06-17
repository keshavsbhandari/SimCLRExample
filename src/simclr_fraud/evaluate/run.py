"""Run evaluation: collect predictions, plot curves, log to W&B."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from simclr_fraud.evaluate.plots import (
    find_best_f1_threshold,
    plot_confusion_matrix,
    plot_pr_curve,
    plot_roc_curve,
    plot_score_distribution,
    plot_threshold_sweep,
    threshold_metrics_table,
)
from simclr_fraud.models.classifier import FraudClassifier
from simclr_fraud.paths import eval_dir

log = logging.getLogger(__name__)

DEFAULT_EVAL = {
    "log_wandb": True,
    "save_plots": True,
    "thresholds": [0.5, 0.7, 0.9, 0.95, 0.99, 0.995],
    "find_best_f1_on": "val",
}


def _eval_settings(cfg: DictConfig) -> DictConfig:
    """Merge experiment eval config with defaults."""
    if cfg.get("eval"):
        return OmegaConf.merge(OmegaConf.create(DEFAULT_EVAL), cfg.eval)
    return OmegaConf.create(DEFAULT_EVAL)


@torch.no_grad()
def collect_predictions(
    model: FraudClassifier,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Run model over a dataloader and collect probabilities and labels.

    Args:
        model: Trained fraud classifier.
        dataloader: Batches of ``(x, y)``.
        device: Torch device for inference.

    Returns:
        Tuple ``(y_score, y_true)`` as numpy arrays.
    """
    model.eval()
    probs_list: list[torch.Tensor] = []
    targets_list: list[torch.Tensor] = []
    for x, y in dataloader:
        x = x.to(device)
        logits = model(x)
        probs_list.append(torch.sigmoid(logits).cpu())
        targets_list.append(y.cpu())
    y_score = torch.cat(probs_list).numpy()
    y_true = torch.cat(targets_list).numpy().astype(int)
    return y_score, y_true


def _save_fig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _log_fig_to_wandb(fig: plt.Figure, key: str, wandb_run: Any) -> None:
    wandb_run.log({key: wandb.Image(fig)})
    plt.close(fig)


def _marker_thresholds(
    eval_cfg: DictConfig,
    y_true_val: np.ndarray | None,
    y_score_val: np.ndarray | None,
) -> list[float]:
    markers = [float(t) for t in eval_cfg.thresholds]
    if eval_cfg.get("find_best_f1_on") == "val" and y_true_val is not None and y_score_val is not None:
        best_t, best_f1 = find_best_f1_threshold(y_true_val, y_score_val)
        if best_t not in markers:
            markers.append(best_t)
        log.info("Best F1 on val: threshold=%.4f f1=%.4f", best_t, best_f1)
    return sorted(set(markers))


def run_evaluation(
    cfg: DictConfig,
    model: FraudClassifier,
    test_loader: DataLoader,
    val_loader: DataLoader | None = None,
    wandb_run: Any | None = None,
) -> dict[str, Any]:
    """Evaluate on test set, save plots locally, and optionally log to W&B.

    Generates ROC/PR curves, threshold sweep, score distribution, and confusion
    matrices. Writes artifacts to ``outputs/{name}/eval/``.

    Args:
        cfg: Experiment config with optional ``eval`` section.
        model: Trained ``FraudClassifier``.
        test_loader: Held-out test DataLoader.
        val_loader: Optional validation loader for best-F1 threshold selection.
        wandb_run: Active W&B run for logging; skipped if ``None``.

    Returns:
        Summary dict with ``test_roc_auc``, ``test_pr_auc``, fraud rate, and
        optional ``val_best_f1_threshold`` / ``val_best_f1``.
    """
    eval_cfg = _eval_settings(cfg)
    out_dir = eval_dir(cfg.name)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    y_score_test, y_true_test = collect_predictions(model, test_loader, device)
    y_score_val, y_true_val = (None, None)
    if val_loader is not None:
        y_score_val, y_true_val = collect_predictions(model, val_loader, device)

    marker_thresholds = _marker_thresholds(eval_cfg, y_true_val, y_score_val)

    roc_auc = float(roc_auc_score(y_true_test, y_score_test))
    pr_auc = float(average_precision_score(y_true_test, y_score_test))
    threshold_df = threshold_metrics_table(y_true_test, y_score_test, marker_thresholds)

    summary = {
        "test_roc_auc": roc_auc,
        "test_pr_auc": pr_auc,
        "fraud_rate": float(y_true_test.mean()),
        "n_test": len(y_true_test),
    }
    if y_score_val is not None and y_true_val is not None:
        best_t, best_f1 = find_best_f1_threshold(y_true_val, y_score_val)
        summary["val_best_f1_threshold"] = best_t
        summary["val_best_f1"] = best_f1

    threshold_df.to_csv(out_dir / "threshold_metrics.csv", index=False)
    log.info("Saved threshold metrics to %s", out_dir / "threshold_metrics.csv")

    should_log_wandb = eval_cfg.log_wandb and wandb_run is not None

    plots: dict[str, plt.Figure] = {
        "roc_curve": plot_roc_curve(y_true_test, y_score_test, marker_thresholds),
        "pr_curve": plot_pr_curve(y_true_test, y_score_test, marker_thresholds),
        "threshold_sweep": plot_threshold_sweep(threshold_df),
        "score_distribution": plot_score_distribution(y_true_test, y_score_test),
    }

    default_t = 0.5 if 0.5 in marker_thresholds else marker_thresholds[0]
    plots[f"confusion_matrix_{default_t}"] = plot_confusion_matrix(
        y_true_test, y_score_test, default_t
    )
    if "val_best_f1_threshold" in summary:
        best_t = summary["val_best_f1_threshold"]
        plots["confusion_matrix_best_f1"] = plot_confusion_matrix(
            y_true_test,
            y_score_test,
            best_t,
            title=f"Confusion Matrix (best val F1, θ={best_t:.4f})",
        )

    saved_paths: dict[str, Path] = {}
    for name, fig in plots.items():
        if eval_cfg.save_plots:
            path = out_dir / f"{name}.png"
            _save_fig(fig, path)
            saved_paths[name] = path
        elif should_log_wandb:
            _log_fig_to_wandb(fig, f"eval/{name}", wandb_run)
        else:
            plt.close(fig)

    if eval_cfg.save_plots:
        log.info("Saved evaluation plots to %s", out_dir)

    if should_log_wandb and eval_cfg.save_plots:
        for name, path in saved_paths.items():
            wandb_run.log({f"eval/{name}": wandb.Image(str(path))})

    if should_log_wandb:
        wandb_run.log({f"eval/{k}": v for k, v in summary.items()})
        wandb_run.log({"eval/threshold_metrics": wandb.Table(dataframe=threshold_df)})
        log.info("Logged evaluation artifacts to W&B")

    return summary
