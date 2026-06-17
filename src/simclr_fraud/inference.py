"""Load trained artifacts for evaluation and inference."""

from __future__ import annotations

import logging

import torch
from omegaconf import DictConfig

from simclr_fraud.models.classifier import FraudClassifier
from simclr_fraud.models.encoder import TabularEncoder
from simclr_fraud.paths import classifier_path

log = logging.getLogger(__name__)


def build_classifier(cfg: DictConfig, input_dim: int) -> FraudClassifier:
    encoder = TabularEncoder(
        input_dim=input_dim,
        hidden_dim=cfg.model.hidden_dim,
        embedding_dim=cfg.model.embedding_dim,
        dropout=cfg.model.dropout,
    )
    return FraudClassifier(
        encoder=encoder,
        embedding_dim=cfg.model.embedding_dim,
        hidden_dim=cfg.model.classifier.hidden_dim,
        dropout=cfg.model.dropout,
    )


def load_classifier(cfg: DictConfig, input_dim: int) -> FraudClassifier:
    """Load a saved classifier from outputs/{name}/finetune/classifier.pt."""
    path = classifier_path(cfg.name)
    if not path.exists():
        raise FileNotFoundError(
            f"No classifier checkpoint at {path}. Run finetune first or check experiment name."
        )

    model = build_classifier(cfg, input_dim)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    log.info("Loaded classifier from %s", path)
    return model
