import torch
import torch.nn.functional as F


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    """NT-Xent / InfoNCE loss for SimCLR contrastive learning."""
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
    """Diagnostic similarity metrics for SimCLR pretraining."""
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
