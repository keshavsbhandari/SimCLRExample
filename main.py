import logging

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig

from simclr_fraud.train.pipeline import run_pipeline


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    load_dotenv()
    run_pipeline(cfg)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
