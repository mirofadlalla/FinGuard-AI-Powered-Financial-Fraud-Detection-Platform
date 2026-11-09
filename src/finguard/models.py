import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, d, p):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, d), nn.BatchNorm1d(d), nn.ReLU(), nn.Dropout(p), nn.Linear(d, d), nn.BatchNorm1d(d))
        self.act = nn.ReLU()
    def forward(self, x): return self.act(x + self.net(x))


class FraudMLP(nn.Module):
    """Tabular MLP. n_blocks=0 -> plain MLP (use for the residual ablation)."""
    def __init__(self, n_features, hidden=128, n_blocks=2, dropout=0.2):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(n_features, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout))
        self.blocks = nn.Sequential(*[ResBlock(hidden, dropout) for _ in range(n_blocks)])
        self.head = nn.Linear(hidden, 1)  # raw logit -> pair with BCEWithLogits / Focal
    def forward(self, x): return self.head(self.blocks(self.inp(x))).squeeze(-1)
