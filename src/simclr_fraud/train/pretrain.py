"""SimCLR contrastive pretraining stage."""

import logging

import torch
from omegaconf import DictConfig

from simclr_fraud.data import ExperimentData
from simclr_fraud.models.lit_simclr import LitSimCLR
from simclr_fraud.paths import encoder_path, pretrain_dir
from simclr_fraud.train.utils import build_trainer, build_wandb_logger, load_best_checkpoint

log = logging.getLogger(__name__)


def run_pretrain(cfg: DictConfig, data: ExperimentData) -> LitSimCLR:
    """Train SimCLR with NT-Xent loss and export the encoder weights.

    Fits ``LitSimCLR`` on ``data.simclr_train`` / ``data.simclr_val``, restores
    the best checkpoint (by ``cfg.pretrain.monitor``), and saves encoder-only
    weights to ``outputs/{name}/pretrain/encoder.pt``.

    Args:
        cfg: Experiment config with ``model``, ``pretrain``, and ``wandb`` sections.
        data: Prepared loaders and feature metadata from ``prepare_data``.

    Returns:
        Trained ``LitSimCLR`` module with best weights loaded.
    """
    model = LitSimCLR(
        input_dim=data.input_dim,
        model_cfg=cfg.model,
        pretrain_cfg=cfg.pretrain,
    )

    logger = build_wandb_logger(cfg, "pretrain")
    trainer, checkpoint_cb = build_trainer(
        cfg=cfg,
        stage_cfg=cfg.pretrain,
        checkpoint_dir=pretrain_dir(cfg.name) / "checkpoints",
        monitor=cfg.pretrain.monitor,
        mode="min",
        logger=logger,
    )

    log.info("Starting SimCLR pretraining for experiment %s", cfg.name)
    trainer.fit(
        model,
        train_dataloaders=data.simclr_train,
        val_dataloaders=data.simclr_val,
    )

    load_best_checkpoint(model, checkpoint_cb)

    out_path = encoder_path(cfg.name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.encoder.state_dict(), out_path)
    log.info("Saved encoder to %s", out_path)

    return model
