"""Batch scoring on new CSV rows using saved experiment artifacts."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import cast

import joblib
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

from simclr_fraud.inference import load_classifier
from simclr_fraud.paths import preprocessor_path, project_root, resolved_config_path

log = logging.getLogger(__name__)


def load_experiment_config(name: str) -> DictConfig:
    path = resolved_config_path(name)
    if not path.is_file():
        raise FileNotFoundError(
            f"No resolved config at {path}. Run training for experiment {name!r} first."
        )
    return cast(DictConfig, OmegaConf.load(path))


def load_preprocessor(name: str):
    path = preprocessor_path(name)
    if not path.is_file():
        raise FileNotFoundError(
            f"No preprocessor at {path}. Run training for experiment {name!r} first."
        )
    log.info("Loaded preprocessor from %s", path)
    return joblib.load(path)


def _prepare_features(df: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    drop_cols = [c for c in cfg.data.drop_cols if c in df.columns]
    return df.drop(columns=drop_cols)


@torch.no_grad()
def score_dataframe(
    cfg: DictConfig,
    df: pd.DataFrame,
    *,
    device: torch.device | None = None,
) -> np.ndarray:
    """Return fraud probabilities for rows in *df*."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    preprocessor = load_preprocessor(cfg.name)
    features = _prepare_features(df, cfg)
    X = np.asarray(preprocessor.transform(features), dtype=np.float32)
    input_dim = X.shape[1]

    model = load_classifier(cfg, input_dim).to(device)
    model.eval()

    x_tensor = torch.tensor(X, device=device)
    logits = model(x_tensor)
    return torch.sigmoid(logits).cpu().numpy()


def predict_csv(
    cfg: DictConfig,
    input_path: Path,
    output_path: Path,
) -> None:
    df = pd.read_csv(input_path)
    probs = score_dataframe(cfg, df)

    out = df.copy()
    out["fraud_probability"] = probs
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    log.info("Wrote %d scores to %s", len(out), output_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Score a CSV with a trained fraud classifier.")
    parser.add_argument(
        "--experiment",
        required=True,
        help="Experiment name (must have outputs/{name}/ artifacts).",
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Input CSV with PaySim-style columns.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output CSV path (adds fraud_probability column).",
    )
    args = parser.parse_args()

    input_path = args.input if args.input.is_absolute() else project_root() / args.input
    output_path = args.output if args.output.is_absolute() else project_root() / args.output

    cfg = load_experiment_config(args.experiment)
    predict_csv(cfg, input_path, output_path)


if __name__ == "__main__":
    main()
