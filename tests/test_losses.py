import torch

from simclr_fraud.models.losses import nt_xent_loss, simclr_similarity_metrics


def test_nt_xent_loss_is_scalar():
    z1 = torch.randn(8, 32)
    z2 = torch.randn(8, 32)
    loss = nt_xent_loss(z1, z2)
    assert loss.ndim == 0
    assert loss.item() > 0


def test_simclr_similarity_metrics_keys():
    z1 = torch.randn(4, 16)
    z2 = torch.randn(4, 16)
    metrics = simclr_similarity_metrics(z1, z2)
    assert set(metrics) == {"positive_similarity", "negative_similarity", "similarity_gap"}
