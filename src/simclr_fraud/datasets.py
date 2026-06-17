import torch
from torch.utils.data import Dataset

AugmentMode = str  # "typed" | "uniform"


class SimCLRTabularDataset(Dataset):
    """Self-supervised dataset returning two augmented views per transaction."""

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
        x = self.X[idx]
        return self.augment(x), self.augment(x)


class FraudDataset(Dataset):
    """Supervised fraud classification dataset."""

    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        self.X = X
        self.y = y

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]
