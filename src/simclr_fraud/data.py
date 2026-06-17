"""PaySim download, sklearn preprocessing, and PyTorch DataLoader construction.

Pipeline: download CSV → stratified train/val/test split → fit preprocessor on
train only → optional stratified subsample → build SimCLR and supervised loaders.

See ``docs/PIPELINE.md`` for the full data stage in context.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import joblib
import kagglehub
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader

from simclr_fraud.datasets import FraudDataset, SimCLRTabularDataset
from simclr_fraud.paths import preprocessor_path, project_root

log = logging.getLogger(__name__)


@dataclass
class ExperimentData:
    """All tensors and DataLoaders needed for pretrain and finetune stages.

    Attributes:
        input_dim: Preprocessed feature dimension after one-hot encoding.
        pos_weight: ``num_negative / num_positive`` for BCE loss weighting.
        numeric_indices: Column indices for typed SimCLR augmentation.
        categorical_slices: One-hot block slices for typed augmentation.
        simclr_train: Augmented pairs for contrastive pretraining (shuffled).
        simclr_val: Augmented pairs for pretrain validation.
        train: Supervised train loader for classifier finetune.
        val: Supervised validation loader.
        test: Held-out test loader (used after finetune).
    """

    input_dim: int
    pos_weight: torch.Tensor
    numeric_indices: list[int]
    categorical_slices: list[tuple[int, int]]
    simclr_train: DataLoader
    simclr_val: DataLoader
    train: DataLoader
    val: DataLoader
    test: DataLoader


def _resolve_path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else project_root() / p


def stratified_tensor_subset(
    X: torch.Tensor,
    y: torch.Tensor,
    fraction: float = 1.0,
    random_state: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select a stratified subset of tensors, preserving fraud ratio.

    Args:
        X: Feature tensor of shape ``[N, D]``.
        y: Label tensor of shape ``[N]``.
        fraction: Fraction of rows to keep in ``(0, 1]``. Returns unchanged if ``>= 1``.
        random_state: Seed for sklearn stratified sampling.

    Returns:
        Subset ``(X_sub, y_sub)`` with approximately the same fraud rate.

    Raises:
        ValueError: If ``fraction`` is not in ``(0, 1]``.
    """
    if fraction >= 1.0:
        return X, y

    if not 0.0 < fraction < 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")

    indices = torch.arange(len(y)).numpy()
    subset_indices, _ = train_test_split(
        indices,
        train_size=fraction,
        random_state=random_state,
        stratify=y.numpy(),
    )
    subset_indices = torch.tensor(subset_indices, dtype=torch.long)

    X_sub = X[subset_indices]
    y_sub = y[subset_indices]

    log.info(
        "Stratified subset: %d -> %d (fraction=%.4f, fraud ratio %.6f -> %.6f)",
        len(y),
        len(y_sub),
        fraction,
        y.float().mean().item(),
        y_sub.float().mean().item(),
    )
    return X_sub, y_sub


def ensure_csv(cfg: DictConfig, force: bool = False) -> Path:
    """Download PaySim CSV to ``cfg.data.csv_path`` if missing.

    Uses ``kagglehub`` with ``cfg.data.kaggle_dataset``. Caches locally after
    first download unless ``force=True``.

    Args:
        cfg: Experiment config with ``data.csv_path`` and ``data.kaggle_dataset``.
        force: If True, delete cached CSV and re-download.

    Returns:
        Absolute path to the local CSV file.

    Raises:
        FileNotFoundError: If the Kaggle dataset contains no CSV file.
    """
    csv_path = _resolve_path(cfg.data.csv_path)
    if force and csv_path.exists():
        csv_path.unlink()
        log.info("Removed cached CSV (force_download=true)")

    if csv_path.exists():
        log.info("Using cached CSV: %s", csv_path)
        return csv_path

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading dataset: %s", cfg.data.kaggle_dataset)
    dataset_path = Path(kagglehub.dataset_download(cfg.data.kaggle_dataset))
    source_csv = next(dataset_path.glob("*.csv"), None)
    if source_csv is None:
        raise FileNotFoundError(f"No CSV file found in {dataset_path}")

    shutil.copy2(source_csv, csv_path)
    log.info("Saved CSV to %s", csv_path)
    return csv_path


def _stratified_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float,
    val_test_ratio: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=val_test_ratio, random_state=seed, stratify=y_temp,
    )
    return (
        cast(pd.DataFrame, X_train),
        cast(pd.DataFrame, X_val),
        cast(pd.DataFrame, X_test),
        cast(pd.Series, y_train),
        cast(pd.Series, y_val),
        cast(pd.Series, y_test),
    )


def _build_preprocessor(
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_cols,
            ),
        ]
    )


def _feature_layout(preprocessor: ColumnTransformer) -> tuple[list[int], list[tuple[int, int]]]:
    numeric_indices: list[int] = []
    categorical_slices: list[tuple[int, int]] = []
    offset = 0

    for name, transformer, cols in preprocessor.transformers_:
        if name == "remainder" and transformer == "drop":
            continue
        if name == "num":
            n_features = len(cols)
        elif name == "cat":
            onehot = transformer.named_steps["onehot"]
            n_features = sum(len(c) for c in onehot.categories_)
        else:
            raise ValueError(f"Unknown transformer block: {name!r}")

        if name == "num":
            numeric_indices.extend(range(offset, offset + n_features))
        elif name == "cat":
            categorical_slices.append((offset, offset + n_features))
        offset += n_features

    return numeric_indices, categorical_slices


def _to_tensors(
    X_arr: np.ndarray,
    y_series: pd.Series,
    fraction: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    X_t = torch.tensor(X_arr.astype(np.float32))
    y_t = torch.tensor(y_series.values.astype(np.float32))
    return stratified_tensor_subset(X_t, y_t, fraction, seed)


def _load_splits(cfg: DictConfig) -> tuple[dict[str, torch.Tensor], dict]:
    csv_path = ensure_csv(cfg, force=cfg.get("force_download", False))
    df = pd.read_csv(csv_path)
    log.info("Loaded PaySim shape: %s", df.shape)

    y = df[cfg.data.target_col].astype(int)
    X = df.drop(columns=list(cfg.data.drop_cols))

    X_train, X_val, X_test, y_train, y_val, y_test = _stratified_split(
        X,
        y,
        test_size=cfg.data.split.test_size,
        val_test_ratio=cfg.data.split.val_test_ratio,
        seed=cfg.data.split.seed,
    )

    numeric_cols = X_train.select_dtypes(
        include=["int64", "float64", "int32", "float32"]
    ).columns.tolist()
    categorical_cols = X_train.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    preprocessor = _build_preprocessor(numeric_cols, categorical_cols)
    X_train_p = np.asarray(preprocessor.fit_transform(X_train))
    X_val_p = np.asarray(preprocessor.transform(X_val))
    X_test_p = np.asarray(preprocessor.transform(X_test))

    preproc_out = preprocessor_path(cfg.name)
    preproc_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, preproc_out)
    log.info("Saved preprocessor to %s", preproc_out)

    numeric_indices, categorical_slices = _feature_layout(preprocessor)
    fraction = cfg.data.fraction
    seed = cfg.data.fraction_seed

    X_train_t, y_train_t = _to_tensors(X_train_p, y_train, fraction, seed)
    X_val_t, y_val_t = _to_tensors(X_val_p, y_val, fraction, seed)
    X_test_t, y_test_t = _to_tensors(X_test_p, y_test, fraction, seed)

    tensors = {
        "X_train": X_train_t,
        "y_train": y_train_t,
        "X_val": X_val_t,
        "y_val": y_val_t,
        "X_test": X_test_t,
        "y_test": y_test_t,
    }
    feature_info = {
        "input_dim": X_train_t.shape[1],
        "numeric_indices": numeric_indices,
        "categorical_slices": categorical_slices,
    }
    return tensors, feature_info


def _dataloader_kwargs(cfg: DictConfig) -> dict:
    dl = cfg.data.dataloader
    kwargs: dict = {
        "num_workers": dl.num_workers,
        "pin_memory": dl.pin_memory,
    }
    if dl.num_workers > 0:
        if dl.get("persistent_workers", False):
            kwargs["persistent_workers"] = True
        prefetch_factor = dl.get("prefetch_factor")
        if prefetch_factor is not None:
            kwargs["prefetch_factor"] = prefetch_factor
    return kwargs


def _simclr_dataset(cfg: DictConfig, X: torch.Tensor, feature_info: dict) -> SimCLRTabularDataset:
    aug = cfg.data.augmentation
    category_dropout = aug.get("category_dropout", aug.feature_dropout)
    return SimCLRTabularDataset(
        X,
        feature_dropout=aug.feature_dropout,
        noise_std=aug.noise_std,
        mode=aug.get("mode", "typed"),
        numeric_indices=feature_info["numeric_indices"],
        categorical_slices=feature_info["categorical_slices"],
        category_dropout=category_dropout,
    )


def prepare_data(cfg: DictConfig) -> ExperimentData:
    """Load PaySim, preprocess, and build all training DataLoaders.

    Fits the sklearn preprocessor on the training split only, saves
    ``preprocessor.joblib`` to the experiment output dir, and applies
    ``cfg.data.fraction`` subsampling per split.

    Args:
        cfg: Full Hydra experiment config (``data``, ``name``, batch sizes).

    Returns:
        ``ExperimentData`` with SimCLR and supervised loaders ready for training.
    """
    tensors, feature_info = _load_splits(cfg)

    X_train = tensors["X_train"]
    y_train = tensors["y_train"]
    X_val = tensors["X_val"]
    y_val = tensors["y_val"]
    X_test = tensors["X_test"]
    y_test = tensors["y_test"]

    num_positive = y_train.sum()
    num_negative = len(y_train) - num_positive
    pos_weight = num_negative / num_positive

    dl_kw = _dataloader_kwargs(cfg)
    bs = cfg.data.batch_size

    log.info("Data ready (fraction=%s)", cfg.data.fraction)
    log.info("Train: %s fraud ratio: %.6f", tuple(X_train.shape), y_train.mean().item())
    log.info("Val:   %s fraud ratio: %.6f", tuple(X_val.shape), y_val.mean().item())
    log.info("Test:  %s fraud ratio: %.6f", tuple(X_test.shape), y_test.mean().item())
    log.info("pos_weight: %.4f", pos_weight.item())

    return ExperimentData(
        input_dim=feature_info["input_dim"],
        pos_weight=pos_weight,
        numeric_indices=feature_info["numeric_indices"],
        categorical_slices=feature_info["categorical_slices"],
        simclr_train=DataLoader(
            _simclr_dataset(cfg, X_train, feature_info),
            batch_size=bs.pretrain,
            shuffle=True,
            drop_last=cfg.data.dataloader.drop_last_simclr,
            **dl_kw,
        ),
        simclr_val=DataLoader(
            _simclr_dataset(cfg, X_val, feature_info),
            batch_size=bs.pretrain,
            shuffle=False,
            drop_last=cfg.data.dataloader.drop_last_simclr,
            **dl_kw,
        ),
        train=DataLoader(
            FraudDataset(X_train, y_train),
            batch_size=bs.finetune_train,
            shuffle=True,
            **dl_kw,
        ),
        val=DataLoader(
            FraudDataset(X_val, y_val),
            batch_size=bs.finetune_eval,
            shuffle=False,
            **dl_kw,
        ),
        test=DataLoader(
            FraudDataset(X_test, y_test),
            batch_size=bs.finetune_eval,
            shuffle=False,
            **dl_kw,
        ),
    )
