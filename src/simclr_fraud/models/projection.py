import torch.nn as nn


class ProjectionHead(nn.Module):
    """SimCLR projection head applied to encoder embeddings."""

    def __init__(self, embedding_dim: int = 128, projection_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, projection_dim),
        )

    def forward(self, h):
        return self.net(h)
