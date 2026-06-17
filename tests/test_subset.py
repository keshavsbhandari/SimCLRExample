import torch

from simclr_fraud.data import stratified_tensor_subset


def test_fraction_one_returns_unchanged():
    X = torch.randn(100, 5)
    y = torch.cat([torch.zeros(90), torch.ones(10)])
    X_out, y_out = stratified_tensor_subset(X, y, fraction=1.0)
    assert X_out.shape == X.shape
    assert y_out.shape == y.shape


def test_stratified_subset_preserves_fraud_ratio():
    torch.manual_seed(0)
    y = torch.cat([torch.zeros(9000), torch.ones(1000)])
    X = torch.randn(len(y), 8)
    original_ratio = y.float().mean().item()

    X_sub, y_sub = stratified_tensor_subset(X, y, fraction=0.1, random_state=42)
    subset_ratio = y_sub.float().mean().item()

    assert len(y_sub) < len(y)
    assert abs(subset_ratio - original_ratio) < 0.005


def test_invalid_fraction_raises():
    X = torch.randn(10, 3)
    y = torch.zeros(10)
    try:
        stratified_tensor_subset(X, y, fraction=0.0)
        raised = False
    except ValueError:
        raised = True
    assert raised
