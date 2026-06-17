"""PyTorch datasets for SimCLR pretraining and supervised fraud classification."""

import torch
from torch.utils.data import Dataset

AugmentMode = str  # "typed" | "uniform"


class SimCLRTabularDataset(Dataset):
    """Self-supervised dataset returning two augmented views per transaction.

  Each ``__getitem__`` call independently augments the same row twice, producing
  a positive pair ``(x1, x2)`` for NT-Xent contrastive learning. Labels are
  not used during pretraining.

  Augmentation modes:

  - ``uniform``: Feature dropout + Gaussian noise on all columns.
  - ``typed``: Noise/dropout on numeric columns; whole one-hot category blocks
    zeroed with ``category_dropout`` probability.

  Args:
      X: Preprocessed feature tensor of shape ``[N, D]``.
      feature_dropout: Probability of zeroing a numeric feature (or all features
          in uniform mode).
      noise_std: Standard deviation of Gaussian noise added to numerics.
      mode: ``"typed"`` or ``"uniform"``.
      numeric_indices: Column indices of scaled numeric features (typed mode).
      categorical_slices: ``(start, end)`` slices for one-hot blocks (typed mode).
      category_dropout: Probability of zeroing an entire categorical block.
          Defaults to ``feature_dropout`` when omitted.
  """

    def __init__(
        self,
        X: torch.Tensor,
        feature_dropout: float = 0.15,
        noise_std: float = 0.05,
        mode: AugmentMode = "typed",
        numeric_indices: list[int] | None = None,
        categorical_slices: list[tuple[int, int]] | None = None,
        category_dropout: float | None = None,
    ):
        self.X = X
        self.feature_dropout = feature_dropout
        self.noise_std = noise_std
        self.mode = mode
        self.numeric_indices = numeric_indices or []
        self.categorical_slices = categorical_slices or []
        self.category_dropout = (
            feature_dropout if category_dropout is None else category_dropout
        )

        if self.mode == "typed" and not self.numeric_indices and not self.categorical_slices:
            raise ValueError(
                "typed augmentation requires numeric_indices and/or categorical_slices"
            )

    def augment(self, x: torch.Tensor) -> torch.Tensor:
        """Apply one stochastic augmentation to a single feature vector."""
        if self.mode == "uniform":
            return self._augment_uniform(x)
        return self._augment_typed(x)

    def _augment_uniform(self, x: torch.Tensor) -> torch.Tensor:
        x_aug = x.clone()
        mask = torch.rand_like(x_aug) > self.feature_dropout
        x_aug = x_aug * mask
        noise = torch.randn_like(x_aug) * self.noise_std
        return x_aug + noise

    def _augment_typed(self, x: torch.Tensor) -> torch.Tensor:
        x_aug = x.clone()

        if self.numeric_indices:
            num_idx = torch.tensor(self.numeric_indices, dtype=torch.long, device=x.device)
            num_vals = x_aug[num_idx]
            mask = torch.rand_like(num_vals) > self.feature_dropout
            noise = torch.randn_like(num_vals) * self.noise_std
            x_aug[num_idx] = num_vals * mask + noise

        for start, end in self.categorical_slices:
            if torch.rand(1, device=x.device).item() <= self.category_dropout:
                x_aug[start:end] = 0.0

        return x_aug

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return two independent augmented views of transaction ``idx``."""
        x = self.X[idx]
        return self.augment(x), self.augment(x)


class FraudDataset(Dataset):
    """Supervised dataset of preprocessed transactions and fraud labels.

    Args:
        X: Feature tensor of shape ``[N, D]``.
        y: Binary labels of shape ``[N]`` (0 = legitimate, 1 = fraud).
    """

    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        self.X = X
        self.y = y

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]
