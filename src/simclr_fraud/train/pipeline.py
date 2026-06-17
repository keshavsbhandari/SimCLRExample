import logging

import lightning as L
from omegaconf import DictConfig

from simclr_fraud.config import ensure_experiment_dirs, save_resolved_config, validate_config
from simclr_fraud.data import prepare_data
from simclr_fraud.paths import control_path
from simclr_fraud.train.control import ensure_control_file, should_skip_finetune
from simclr_fraud.train.eval_only import run_eval_only
from simclr_fraud.train.finetune import run_finetune
from simclr_fraud.train.pretrain import run_pretrain

log = logging.getLogger(__name__)


def run_pipeline(cfg: DictConfig) -> None:
    """Run pretrain and/or finetune according to *cfg*."""
    validate_config(cfg)
    ensure_experiment_dirs(cfg)
    ensure_control_file(cfg.name)

    seed = int(cfg.get("seed", 42))
    L.seed_everything(seed, workers=True)
    log.info("Random seed: %d", seed)

    config_path = save_resolved_config(cfg)
    log.info("Experiment: %s", cfg.name)
    log.info("Resolved config saved to %s", config_path)
    log.info("Runtime control: %s", control_path(cfg.name))
    log.info(
        "run_pretrain=%s run_finetune=%s run_eval_only=%s",
        cfg.run_pretrain,
        cfg.run_finetune,
        cfg.get("run_eval_only", False),
    )
    log.info("data.fraction=%s", cfg.data.fraction)

    if cfg.get("run_eval_only", False):
        run_eval_only(cfg)
        return

    data = prepare_data(cfg)

    if cfg.run_pretrain:
        run_pretrain(cfg, data)

    if cfg.run_finetune:
        if should_skip_finetune(cfg.name):
            log.info("control.yaml: skip_finetune=true — skipping finetune stage")
        else:
            run_finetune(cfg, data)

    log.info("Pipeline complete for experiment %s", cfg.name)
