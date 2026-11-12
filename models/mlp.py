import torch.nn as nn


class ResBlock(nn.Module):
    """Pre-activation residual block for tabular data.

    Args:
        dim:     Width of the hidden layer (must match the input width for the skip connection).
        dropout: Dropout probability applied after the first activation.
    """

    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(x + self.net(x))


class FraudMLP(nn.Module):
    """Tabular MLP with optional residual blocks for fraud detection.

    Args:
        n_features: Number of input features.
        hidden:     Hidden layer width (default 128).
        n_blocks:   Number of ResBlocks after the input projection.
                    Set to 0 for a plain MLP (useful for the residual ablation study).
        dropout:    Dropout probability used throughout the network.

    Output:
        Raw logit (unbounded scalar). Pair with BCEWithLogitsLoss or FocalLoss during
        training; apply torch.sigmoid() at inference time.
    """

    def __init__(self, n_features: int, hidden: int = 128, n_blocks: int = 2, dropout: float = 0.2):
        super().__init__()
        self.inp = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.Sequential(*[ResBlock(hidden, dropout) for _ in range(n_blocks)])
        self.head = nn.Linear(hidden, 1)  # raw logit → pair with BCEWithLogits / Focal

    def forward(self, x):
        return self.head(self.blocks(self.inp(x))).squeeze(-1)
