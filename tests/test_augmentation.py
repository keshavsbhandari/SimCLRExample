import torch

from simclr_fraud.datasets import SimCLRTabularDataset


def test_typed_augment_leaves_categorical_binary_when_not_dropped():
    # 2 numeric + 3 one-hot categorical
    X = torch.tensor([[1.0, 2.0, 1.0, 0.0, 0.0]], dtype=torch.float32)
    ds = SimCLRTabularDataset(
        X,
        mode="typed",
        numeric_indices=[0, 1],
        categorical_slices=[(2, 5)],
        feature_dropout=0.0,
        noise_std=0.0,
        category_dropout=0.0,
    )
    torch.manual_seed(0)
    out = ds.augment(X[0])
    assert out[2:].tolist() == [1.0, 0.0, 0.0]


def test_typed_augment_zeros_categorical_block_on_dropout():
    X = torch.tensor([[1.0, 2.0, 1.0, 0.0, 0.0]], dtype=torch.float32)
    ds = SimCLRTabularDataset(
        X,
        mode="typed",
        numeric_indices=[0, 1],
        categorical_slices=[(2, 5)],
        feature_dropout=0.0,
        noise_std=0.0,
        category_dropout=1.0,
    )
    out = ds.augment(X[0])
    assert out[2:].tolist() == [0.0, 0.0, 0.0]
    assert out[0].item() == 1.0
    assert out[1].item() == 2.0


def test_typed_augment_adds_noise_to_numeric_only():
    X = torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0]], dtype=torch.float32)
    ds = SimCLRTabularDataset(
        X,
        mode="typed",
        numeric_indices=[0, 1],
        categorical_slices=[(2, 5)],
        feature_dropout=0.0,
        noise_std=0.5,
        category_dropout=0.0,
    )
    torch.manual_seed(1)
    out = ds.augment(X[0])
    assert out[0].item() != 0.0 or out[1].item() != 0.0
    assert out[2:].tolist() == [1.0, 0.0, 0.0]


def test_uniform_mode_unchanged():
    X = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    ds = SimCLRTabularDataset(X, mode="uniform", feature_dropout=0.0, noise_std=0.1)
    torch.manual_seed(0)
    out = ds.augment(X[0])
    assert out.shape == X[0].shape
