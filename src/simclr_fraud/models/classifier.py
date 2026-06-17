import torch.nn as nn

from simclr_fraud.models.encoder import TabularEncoder


class FraudClassifier(nn.Module):
    """Binary fraud classifier built on top of a tabular encoder."""

    def __init__(
        self,
        encoder: TabularEncoder,
        embedding_dim: int = 128,
        hidden_dim: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        h = self.encoder(x)
        return self.classifier(h).squeeze(1)
