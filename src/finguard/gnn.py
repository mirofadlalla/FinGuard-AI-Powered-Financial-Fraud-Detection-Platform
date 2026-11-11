"""Graph neural network for fraud: relational GraphSAGE (1 hop) on a time-causal transaction graph.

How one transaction is scored:
  * its own features go through a linear layer
  * for each relation (same client / same card+address / same device / same email+address)
    we take the previous transactions that share that entity, AVERAGE their features,
    add "how many previous ones were there" and pass it through a linear layer of that relation
  * everything is summed, then a small residual network gives the fraud logit
So a transaction can "see" the behaviour of the entities it touches (fraud rings, shared devices),
which a plain MLP cannot. Neighbours are always EARLIER transactions (no future leakage).
"""
import torch
import torch.nn as nn

from finguard.models import ResBlock


class GraphSAGEFraud(nn.Module):
    def __init__(self, X_all, neighbors, hidden=128, n_blocks=2, dropout=0.2):
        """X_all: features of ALL transactions (train + val + test, in time order).
        neighbors: (rows, relations, k) row numbers of earlier transactions, -1 = none."""
        super().__init__()
        n_features = X_all.shape[1]
        n_relations, self.k = neighbors.shape[1], neighbors.shape[2]

        # big tensors: not parameters and not saved in checkpoints, but they move to the GPU with the model
        self.register_buffer("X_all", torch.as_tensor(X_all, dtype=torch.float32), persistent=False)
        self.register_buffer("neighbors", torch.as_tensor(neighbors, dtype=torch.int32), persistent=False)

        self.self_layer = nn.Linear(n_features, hidden)
        self.relation_layers = nn.ModuleList([nn.Linear(n_features + 1, hidden) for _ in range(n_relations)])
        self.after_sum = nn.Sequential(nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout))
        self.blocks = nn.Sequential(*[ResBlock(hidden, dropout) for _ in range(n_blocks)])
        self.head = nn.Linear(hidden, 1)

    def forward(self, node_ids):
        """node_ids: row numbers of the transactions to score -> fraud logits."""
        node_ids = node_ids.long()
        h = self.self_layer(self.X_all[node_ids])                      # (batch, hidden)

        neigh = self.neighbors[node_ids].long()                        # (batch, relations, k)
        for r, layer in enumerate(self.relation_layers):
            ids = neigh[:, r, :]                                       # (batch, k)
            valid = (ids >= 0).unsqueeze(-1).float()                   # (batch, k, 1)
            feats = self.X_all[ids.clamp(min=0)] * valid               # (batch, k, features)
            count = valid.sum(dim=1)                                   # (batch, 1) how many neighbours
            mean = feats.sum(dim=1) / count.clamp(min=1)               # average of the neighbours حساب متوسط Features الجيران
            
            # هنا بنعمل Concatenation بين:
            # - mean: متوسط Features الجيران.
            # - count / self.k: نسبة عدد الجيران للحد الأقصى.
            h = h + layer(torch.cat([mean, count / self.k], dim=1)) 

        return self.head(self.blocks(self.after_sum(h))).squeeze(-1)
