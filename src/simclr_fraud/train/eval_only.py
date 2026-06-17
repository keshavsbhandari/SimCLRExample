"""Re-run evaluation on a saved classifier without training."""

from __future__ import annotations

import json
import logging

from omegaconf import DictConfig

from simclr_fraud.data import prepare_data
from simclr_fraud.evaluate.run import run_evaluation
from simclr_fraud.inference import load_classifier
from simclr_fraud.paths import eval_dir
from simclr_fraud.train.utils import build_wandb_logger

log = logging.getLogger(__name__)


def run_eval_only(cfg: DictConfig) -> None:
    """Load saved classifier, run test metrics and evaluation plots.

    Used when ``run_eval_only=true`` in config — skips pretrain and finetune,
    reloads ``classifier.pt`` and the preprocessor, and writes eval artifacts.

    Args:
        cfg: Experiment config with ``name`` matching a completed training run.
    """
    data = prepare_data(cfg)
    model = load_classifier(cfg, data.input_dim)

    logger = build_wandb_logger(cfg, "eval")
    wandb_run = None
    if logger is not None and hasattr(logger, "experiment"):
        wandb_run = logger.experiment

    log.info("Running eval-only for experiment %s", cfg.name)
    summary = run_evaluation(
        cfg=cfg,
        model=model,
        test_loader=data.test,
        val_loader=data.val,
        wandb_run=wandb_run,
    )

    metrics_path = eval_dir(cfg.name) / "metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(summary, indent=2))
    log.info("Saved eval summary to %s", metrics_path)
    log.info("Eval-only complete for experiment %s", cfg.name)
