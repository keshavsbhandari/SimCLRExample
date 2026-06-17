import json
import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig

from simclr_fraud.train.control import RuntimeControlCallback

log = logging.getLogger(__name__)


def _resolve_wandb_entity(cfg: DictConfig) -> str | None:
    entity = cfg.wandb.entity or os.environ.get("WANDB_ENTITY")
    if entity is not None and not str(entity).strip():
        return None
    return entity


def _wandb_group(cfg: DictConfig) -> str:
    return cfg.wandb.get("experiment_name", cfg.name)


def resolve_wandb_project(cfg: DictConfig) -> tuple[str, str]:
    """Return (project_name, source_label). Config/Hydra wins over WANDB_PROJECT env."""
    yaml_project = cfg.wandb.get("project")
    if yaml_project is not None and str(yaml_project).strip():
        return str(yaml_project).strip(), "config"

    env_project = os.environ.get("WANDB_PROJECT")
    if env_project is not None and str(env_project).strip():
        return str(env_project).strip(), "WANDB_PROJECT env"

    raise ValueError(
        "W&B project not set. Add wandb.project to the experiment config "
        "or set WANDB_PROJECT in .env"
    )


def build_wandb_logger(cfg: DictConfig, stage: str) -> WandbLogger | None:
    if os.environ.get("WANDB_MODE") == "disabled":
        log.info("WANDB_MODE=disabled; skipping W&B logger")
        return None

    if not os.environ.get("WANDB_API_KEY"):
        log.warning("WANDB_API_KEY not set; training without W&B logger")
        return None

    entity = _resolve_wandb_entity(cfg)
    project, project_source = resolve_wandb_project(cfg)
    group = _wandb_group(cfg)
    log.info("W&B project=%r (from %s), group=%r, run=%r", project, project_source, group, f"{group}/{stage}")

    def _try_init(resolved_entity: str | None) -> WandbLogger:
        kwargs: dict = {
            "project": project,
            "name": f"{group}/{stage}",
            "group": group,
            "tags": [group, stage],
            "log_model": cfg.wandb.log_model,
        }
        if resolved_entity:
            kwargs["entity"] = resolved_entity
        logger = WandbLogger(**kwargs)
        _ = logger.experiment
        return logger

    try:
        return _try_init(entity)
    except Exception as exc:
        if entity:
            log.warning(
                "W&B init failed for entity=%r project=%r (%s); retrying with default entity",
                entity,
                project,
                exc,
            )
            try:
                return _try_init(None)
            except Exception as retry_exc:
                log.warning(
                    "Failed to initialize W&B logger (%s). "
                    "Check WANDB_API_KEY, WANDB_ENTITY (your username/team), and WANDB_PROJECT in .env",
                    retry_exc,
                )
                return None

        log.warning(
            "Failed to initialize W&B logger (%s). "
            "Check WANDB_API_KEY, WANDB_ENTITY (your username/team), and WANDB_PROJECT in .env",
            exc,
        )
        return None


def load_best_checkpoint(module: L.LightningModule, checkpoint_cb: ModelCheckpoint) -> None:
    """Restore trainer-saved best weights before export or test."""
    best_path = checkpoint_cb.best_model_path
    if best_path and Path(best_path).is_file():
        checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
        module.load_state_dict(checkpoint["state_dict"])
        log.info("Loaded best checkpoint from %s", best_path)
    else:
        log.warning("No best checkpoint found; using final epoch weights")


def build_trainer(
    cfg: DictConfig,
    stage_cfg: DictConfig,
    checkpoint_dir: Path,
    monitor: str,
    mode: str,
    logger: WandbLogger | None,
) -> tuple[L.Trainer, ModelCheckpoint]:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename="best",
        monitor=monitor,
        mode=mode,
        save_top_k=1,
    )

    callbacks: list = [checkpoint_cb, RuntimeControlCallback(experiment_name=cfg.name)]
    patience = stage_cfg.get("early_stopping_patience")
    if patience is not None and int(patience) > 0:
        callbacks.append(
            EarlyStopping(
                monitor=monitor,
                mode=mode,
                patience=int(patience),
                verbose=True,
            )
        )
        log.info("Early stopping enabled: monitor=%s patience=%d mode=%s", monitor, patience, mode)

    trainer_kwargs: dict = {
        "max_epochs": stage_cfg.max_epochs,
        "accelerator": stage_cfg.accelerator,
        "devices": stage_cfg.devices,
        "precision": stage_cfg.get("precision", "32-true"),
        "logger": logger,
        "callbacks": callbacks,
        "log_every_n_steps": stage_cfg.log_every_n_steps,
        "enable_progress_bar": True,
    }

    if stage_cfg.get("limit_train_batches") is not None:
        trainer_kwargs["limit_train_batches"] = stage_cfg.limit_train_batches
    if stage_cfg.get("limit_val_batches") is not None:
        trainer_kwargs["limit_val_batches"] = stage_cfg.limit_val_batches
    if stage_cfg.get("limit_test_batches") is not None:
        trainer_kwargs["limit_test_batches"] = stage_cfg.limit_test_batches

    return L.Trainer(**trainer_kwargs), checkpoint_cb


def save_metrics(metrics: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {k: float(v) if hasattr(v, "item") else v for k, v in metrics.items()}
    path.write_text(json.dumps(serializable, indent=2))
    log.info("Saved metrics to %s", path)
