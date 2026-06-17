from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def project_root() -> Path:
    return PROJECT_ROOT


def experiment_dir(name: str) -> Path:
    return PROJECT_ROOT / "outputs" / name


def pretrain_dir(name: str) -> Path:
    return experiment_dir(name) / "pretrain"


def finetune_dir(name: str) -> Path:
    return experiment_dir(name) / "finetune"


def eval_dir(name: str) -> Path:
    return experiment_dir(name) / "eval"


def resolved_config_path(name: str) -> Path:
    return experiment_dir(name) / "config_resolved.yaml"


def control_path(name: str) -> Path:
    return experiment_dir(name) / "control.yaml"


def encoder_path(name: str) -> Path:
    return pretrain_dir(name) / "encoder.pt"


def finetune_best_checkpoint(name: str) -> Path:
    return finetune_dir(name) / "checkpoints" / "best.ckpt"


def pretrain_best_checkpoint(name: str) -> Path:
    return pretrain_dir(name) / "checkpoints" / "best.ckpt"


def classifier_path(name: str) -> Path:
    return finetune_dir(name) / "classifier.pt"


def preprocessor_path(name: str) -> Path:
    return experiment_dir(name) / "preprocessor.joblib"
