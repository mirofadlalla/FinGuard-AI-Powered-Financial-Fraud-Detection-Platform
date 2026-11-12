"""Machine Learning & Deep Learning models package for FinGuard."""
from models.mlp import FraudMLP, ResBlock
from models.gnn import GraphSAGEFraud
from models.losses import FocalLoss, make_loss
from models.trainer import fit, predict
from models.torch_data import make_loader

# Convenient aliases
GNN = GraphSAGEFraud
get_loss = make_loss
train = fit

__all__ = [
    "FraudMLP",
    "ResBlock",
    "GraphSAGEFraud",
    "GNN",
    "FocalLoss",
    "make_loss",
    "get_loss",
    "fit",
    "train",
    "predict",
    "make_loader",
]
