"""Supervised fraud classifier finetune stage."""

import logging
from pathlib import Path

import torch
from omegaconf import DictConfig

from simclr_fraud.data import ExperimentData
from simclr_fraud.evaluate.run import run_evaluation
from simclr_fraud.models.classifier import FraudClassifier
from simclr_fraud.models.encoder import TabularEncoder
from simclr_fraud.models.lit_classifier import LitFraudClassifier
from simclr_fraud.paths import classifier_path, encoder_path, eval_dir, finetune_dir
from simclr_fraud.train.utils import build_trainer, build_wandb_logger, load_best_checkpoint, save_metrics

log = logging.getLogger(__name__)


def _resolve_encoder_checkpoint(cfg: DictConfig) -> Path | None:
    """Return path to pretrained encoder weights, if available."""
    if cfg.finetune.encoder_checkpoint:
        return Path(cfg.finetune.encoder_checkpoint)

    path = encoder_path(cfg.name)
    if path.exists():
        return path

    return None


def _build_fraud_model(cfg: DictConfig, input_dim: int) -> FraudClassifier:
    """Construct ``FraudClassifier`` with optional pretrained encoder weights.

    Loads from ``cfg.finetune.encoder_checkpoint`` or default ``encoder.pt``.
    If no checkpoint exists, uses a random encoder (supervised baseline).
    Optionally freezes encoder when ``cfg.finetune.freeze_encoder`` is True.
    """
    encoder = TabularEncoder(
        input_dim=input_dim,
        hidden_dim=cfg.model.hidden_dim,
        embedding_dim=cfg.model.embedding_dim,
        dropout=cfg.model.dropout,
    )

    checkpoint = _resolve_encoder_checkpoint(cfg)
    if checkpoint is not None:
        log.info("Loading encoder weights from %s", checkpoint)
        encoder.load_state_dict(torch.load(checkpoint, weights_only=True))
    else:
        log.info("Training classifier with a randomly initialized encoder")

    model = FraudClassifier(
        encoder=encoder,
        embedding_dim=cfg.model.embedding_dim,
        hidden_dim=cfg.model.classifier.hidden_dim,
        dropout=cfg.model.dropout,
    )

    if cfg.finetune.freeze_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = False
        log.info("Encoder frozen during finetune")

    return model


def run_finetune(cfg: DictConfig, data: ExperimentData) -> LitFraudClassifier:
    """Finetune fraud classifier, test, and run evaluation plots.

    Trains ``LitFraudClassifier`` on supervised loaders, saves ``classifier.pt``,
    runs Lightning ``test``, writes ``eval/metrics.json``, and optionally generates
    ROC/PR plots via ``run_evaluation``.

    Args:
        cfg: Experiment config with ``model``, ``finetune``, ``eval``, and ``wandb``.
        data: Prepared loaders including ``pos_weight`` for class imbalance.

    Returns:
        Trained ``LitFraudClassifier`` with best weights loaded.
    """
    fraud_model = _build_fraud_model(cfg, data.input_dim)
    lit_model = LitFraudClassifier(
        model=fraud_model,
        pos_weight=data.pos_weight,
        finetune_cfg=cfg.finetune,
    )

    logger = build_wandb_logger(cfg, "finetune")
    trainer, checkpoint_cb = build_trainer(
        cfg=cfg,
        stage_cfg=cfg.finetune,
        checkpoint_dir=finetune_dir(cfg.name) / "checkpoints",
        monitor=cfg.finetune.monitor,
        mode="max",
        logger=logger,
    )

    log.info("Starting fraud classifier finetune for experiment %s", cfg.name)
    trainer.fit(
        lit_model,
        train_dataloaders=data.train,
        val_dataloaders=data.val,
    )

    load_best_checkpoint(lit_model, checkpoint_cb)

    ckpt_path = classifier_path(cfg.name)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(lit_model.model.state_dict(), ckpt_path)
    log.info("Saved classifier to %s", ckpt_path)

    test_results = trainer.test(lit_model, dataloaders=data.test)
    if test_results:
        save_metrics(test_results[0], eval_dir(cfg.name) / "metrics.json")

    wandb_run = None
    if logger is not None and hasattr(logger, "experiment"):
        wandb_run = logger.experiment

    eval_cfg = cfg.get("eval")
    run_eval = eval_cfg.get("enabled", True) if eval_cfg is not None else True
    if run_eval:
        log.info("Running evaluation plots for experiment %s", cfg.name)
        run_evaluation(
            cfg=cfg,
            model=lit_model.model,
            test_loader=data.test,
            val_loader=data.val,
            wandb_run=wandb_run,
        )

    return lit_model
