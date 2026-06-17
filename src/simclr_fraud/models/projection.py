"""SimCLR projection head applied during contrastive pretraining only."""

import torch.nn as nn


class ProjectionHead(nn.Module):
    """Two-layer MLP that maps encoder embeddings to projection space ``z``.

    SimCLR applies NT-Xent loss on ``z``, not on ``h``. The projection head is
    discarded after pretraining; only the encoder weights are saved for finetune.

    Args:
        embedding_dim: Input dimension (encoder output size).
        projection_dim: Output dimension used for contrastive loss.
    """

    def __init__(self, embedding_dim: int = 128, projection_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, projection_dim),
        )

    def forward(self, h):
        """Project encoder embeddings into contrastive space.

        Args:
            h: Encoder output of shape ``[B, embedding_dim]``.

        Returns:
            Projections ``z`` of shape ``[B, projection_dim]``.
        """
        return self.net(h)
