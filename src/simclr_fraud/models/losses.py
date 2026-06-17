"""Contrastive loss and diagnostic metrics for SimCLR pretraining."""

import torch
import torch.nn.functional as F


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    """NT-Xent (InfoNCE) contrastive loss for a batch of projection pairs.

    Treats ``(z1[i], z2[i])`` as positive pairs and all other cross-sample pairs
    in the ``2B × 2B`` similarity matrix as negatives. Inputs are L2-normalized
    inside this function.

    Args:
        z1: Projections for augmented view 1, shape ``[B, D]``.
        z2: Projections for augmented view 2, shape ``[B, D]``.
        temperature: Softmax temperature τ; lower values sharpen the distribution.

    Returns:
        Scalar cross-entropy loss over in-batch negatives.
    """
    batch_size = z1.size(0)

    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    z = torch.cat([z1, z2], dim=0)

    sim = torch.matmul(z, z.T) / temperature
    self_mask = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(self_mask, -1e9)

    positive_indices = torch.arange(2 * batch_size, device=z.device)
    positive_indices = (positive_indices + batch_size) % (2 * batch_size)

    return F.cross_entropy(sim, positive_indices)


def simclr_similarity_metrics(z1: torch.Tensor, z2: torch.Tensor) -> dict[str, torch.Tensor]:
    """Diagnostic cosine-similarity metrics for monitoring SimCLR pretraining.

    Useful for tracking whether positive pairs are pulled together and negatives
    pushed apart during training. All inputs are L2-normalized internally.

    Args:
        z1: Projections for view 1, shape ``[B, D]``.
        z2: Projections for view 2, shape ``[B, D]``.

    Returns:
        Dict with keys:

        - ``positive_similarity``: Mean cosine sim between matched pairs (want ↑).
        - ``negative_similarity``: Mean cosine sim between negatives (want ↓).
        - ``similarity_gap``: ``positive - negative`` (want ↑).
    """
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    batch_size = z1.size(0)

    positive_sim = torch.sum(z1 * z2, dim=1).mean()
    z = torch.cat([z1, z2], dim=0)
    sim = torch.matmul(z, z.T)

    self_mask = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)
    positive_mask = torch.zeros_like(self_mask)
    for i in range(batch_size):
        positive_mask[i, i + batch_size] = True
        positive_mask[i + batch_size, i] = True

    negative_mask = ~(self_mask | positive_mask)
    negative_sim = sim[negative_mask].mean()
    similarity_gap = positive_sim - negative_sim

    return {
        "positive_similarity": positive_sim,
        "negative_similarity": negative_sim,
        "similarity_gap": similarity_gap,
    }
