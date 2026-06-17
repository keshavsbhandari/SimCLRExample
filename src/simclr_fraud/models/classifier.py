"""Binary fraud classifier built on a tabular encoder."""

import torch.nn as nn

from simclr_fraud.models.encoder import TabularEncoder


class FraudClassifier(nn.Module):
    """Binary fraud classifier: encoder → MLP head → single logit.

    Used during supervised finetune and inference. The encoder may be loaded
    from SimCLR pretraining or initialized randomly (control baseline).

    Args:
        encoder: Tabular encoder producing transaction embeddings.
        embedding_dim: Size of encoder output (must match encoder).
        hidden_dim: Width of the classifier hidden layer.
        dropout: Dropout after the classifier hidden ReLU.
    """

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
        """Return raw logits for a batch of transactions.

        Args:
            x: Preprocessed features of shape ``[B, input_dim]``.

        Returns:
            Logits of shape ``[B]`` (squeeze applied). Apply sigmoid for probabilities.
        """
        h = self.encoder(x)
        return self.classifier(h).squeeze(1)
