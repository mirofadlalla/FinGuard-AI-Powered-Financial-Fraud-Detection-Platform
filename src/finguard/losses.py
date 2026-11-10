import torch, torch.nn as nn, torch.nn.functional as F


class FocalLoss(nn.Module):
    """Binary focal loss on logits (Lin et al. 2017): down-weights easy negatives."""
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__(); self.alpha, self.gamma = alpha, gamma
    def forward(self, logits, y):
        bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
        pt = torch.exp(-bce)
        a = self.alpha * y + (1 - self.alpha) * (1 - y)
        return (a * (1 - pt) ** self.gamma * bce).mean()


def make_loss(kind: str, pos_weight: float = 1.0, device="cpu"):
    if kind == "bce":      return nn.BCEWithLogitsLoss()
    if kind == "weighted": return nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    if kind == "focal":    return FocalLoss()
    raise ValueError(kind)
