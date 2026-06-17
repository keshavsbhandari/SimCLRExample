from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from simclr_fraud.paths import experiment_dir, resolved_config_path


def validate_config(cfg: DictConfig) -> None:
    if not cfg.get("name"):
        raise ValueError("Config must define 'name' (set in experiment config).")

    fraction = cfg.data.fraction
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"data.fraction must be in (0, 1], got {fraction}")


def save_resolved_config(cfg: DictConfig) -> Path:
    """Write the fully merged config to outputs/{name}/config_resolved.yaml."""
    out_path = resolved_config_path(cfg.name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out_path)
    return out_path


def ensure_experiment_dirs(cfg: DictConfig) -> None:
    """Create output directories for the experiment."""
    base = experiment_dir(cfg.name)
    for sub in ("pretrain/checkpoints", "finetune/checkpoints", "eval"):
        (base / sub).mkdir(parents=True, exist_ok=True)
