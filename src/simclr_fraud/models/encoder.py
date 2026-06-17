"""Tabular MLP encoder used in SimCLR pretraining and downstream fraud classification."""

import torch.nn as nn


class TabularEncoder(nn.Module):
    """Three-layer MLP mapping preprocessed transactions to dense embeddings.

    Architecture: Linear → BatchNorm → ReLU → Dropout (×2 hidden blocks) → Linear.

    The output embedding ``h`` is the representation reused for fraud classification
    after SimCLR pretraining. The projection head (if any) is applied on top of ``h``.

    Args:
        input_dim: Number of features after preprocessing (one-hot + scaled numerics).
        hidden_dim: Width of the two hidden layers.
        embedding_dim: Output dimension of the encoder (``h``).
        dropout: Dropout probability applied after each hidden ReLU.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        embedding_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, x):
        """Encode a batch of preprocessed transactions.

        Args:
            x: Float tensor of shape ``[B, input_dim]``.

        Returns:
            Encoder embeddings ``h`` of shape ``[B, embedding_dim]``.
        """
        return self.net(x)
