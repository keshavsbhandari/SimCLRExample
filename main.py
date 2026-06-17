"""Hydra entry point for the full SimCLR fraud detection training pipeline.

Run from the repository root::

    python main.py experiment=mini

See ``README.md`` and ``docs/PIPELINE.md`` for setup and stage-by-stage flow.
"""

import logging

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig

from simclr_fraud.train.pipeline import run_pipeline


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Load environment variables and run the training pipeline.

    Args:
        cfg: Hydra-composed config (base + experiment YAML + CLI overrides).
    """
    load_dotenv()
    run_pipeline(cfg)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
