"""Canonical paths for project root and per-experiment artifacts.

All experiment outputs live under ``outputs/{name}/``. See ``docs/PIPELINE.md``
for how these paths are used across pretrain, finetune, and eval stages.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def project_root() -> Path:
    """Return the repository root directory."""
    return PROJECT_ROOT


def experiment_dir(name: str) -> Path:
    """Root directory for one experiment: ``outputs/{name}/``."""
    return PROJECT_ROOT / "outputs" / name


def pretrain_dir(name: str) -> Path:
    """SimCLR pretrain artifacts: ``outputs/{name}/pretrain/``."""
    return experiment_dir(name) / "pretrain"


def finetune_dir(name: str) -> Path:
    """Classifier finetune artifacts: ``outputs/{name}/finetune/``."""
    return experiment_dir(name) / "finetune"


def eval_dir(name: str) -> Path:
    """Evaluation plots and metrics: ``outputs/{name}/eval/``."""
    return experiment_dir(name) / "eval"


def resolved_config_path(name: str) -> Path:
    """Frozen merged Hydra config written at pipeline start."""
    return experiment_dir(name) / "config_resolved.yaml"


def control_path(name: str) -> Path:
    """Runtime control file for mid-training stop/skip (see ``train.control``)."""
    return experiment_dir(name) / "control.yaml"


def encoder_path(name: str) -> Path:
    """Encoder state dict exported after SimCLR pretrain."""
    return pretrain_dir(name) / "encoder.pt"


def finetune_best_checkpoint(name: str) -> Path:
    """Best Lightning checkpoint from classifier finetune."""
    return finetune_dir(name) / "checkpoints" / "best.ckpt"


def pretrain_best_checkpoint(name: str) -> Path:
    """Best Lightning checkpoint from SimCLR pretrain."""
    return pretrain_dir(name) / "checkpoints" / "best.ckpt"


def classifier_path(name: str) -> Path:
    """Classifier state dict exported after finetune (used for inference)."""
    return finetune_dir(name) / "classifier.pt"


def preprocessor_path(name: str) -> Path:
    """Sklearn preprocessor fitted on training split only."""
    return experiment_dir(name) / "preprocessor.joblib"
