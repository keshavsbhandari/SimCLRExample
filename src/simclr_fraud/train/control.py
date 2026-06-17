"""Runtime control file and Lightning callback for safe mid-training changes."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, cast

from lightning.pytorch.callbacks import Callback
from omegaconf import DictConfig, OmegaConf

from simclr_fraud.paths import control_path

if TYPE_CHECKING:
    import lightning as L
    from lightning.pytorch.core import LightningModule

log = logging.getLogger(__name__)

DEFAULT_CONTROL = """\
# Runtime controls — edit while training; checked at end of each epoch.
#
# stop: true           → finish this epoch, save checkpoint, exit current stage
# max_epochs: 5        → cap current stage (e.g. lower from 20 to 5 mid-run)
# skip_finetune: true   → after pretrain, skip finetune (checked between stages)

stop: false
skip_finetune: false
max_epochs: null
"""


def read_control(experiment_name: str) -> DictConfig | None:
    """Load ``outputs/{name}/control.yaml`` if it exists."""
    path = control_path(experiment_name)
    if not path.exists():
        return None
    return cast(DictConfig, OmegaConf.load(path))


def ensure_control_file(experiment_name: str) -> DictConfig:
    """Create ``control.yaml`` with defaults if missing; return current contents.

    The control file can be edited during training to stop early, cap epochs,
    or skip finetune after pretrain completes.

    Args:
        experiment_name: Hydra experiment ``name``.

    Returns:
        Parsed control config.
    """
    path = control_path(experiment_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(DEFAULT_CONTROL)
        log.info("Created runtime control file: %s", path)
    control = read_control(experiment_name)
    assert control is not None
    return control


def should_skip_finetune(experiment_name: str) -> bool:
    """Return True if ``control.yaml`` has ``skip_finetune: true``."""
    control = read_control(experiment_name)
    return control is not None and bool(control.get("skip_finetune", False))


class RuntimeControlCallback(Callback):
    """Stop or cap epochs based on ``outputs/{name}/control.yaml``.

    Checked at the end of each training epoch. Supports:

    - ``stop: true`` — finish current epoch, save checkpoint, exit stage.
    - ``max_epochs: N`` — stop once ``N`` epochs have completed.

    Args:
        experiment_name: Hydra experiment ``name`` (locates control file).
    """

    def __init__(self, experiment_name: str) -> None:
        self.experiment_name = experiment_name
        self._control_file = control_path(experiment_name)

    def on_train_start(self, trainer: L.Trainer, pl_module: LightningModule) -> None:
        log.info("Runtime control file: %s", self._control_file)

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: LightningModule) -> None:
        control = read_control(self.experiment_name)
        if control is None:
            return

        if control.get("stop", False):
            log.info("control.yaml: stop=true — stopping gracefully after epoch %d", trainer.current_epoch)
            trainer.should_stop = True
            return

        cap = control.get("max_epochs")
        if cap is not None:
            completed_epochs = trainer.current_epoch + 1
            if completed_epochs >= cap:
                log.info(
                    "control.yaml: max_epochs=%s reached (%d epochs done) — stopping",
                    cap,
                    completed_epochs,
                )
                trainer.should_stop = True
