import torch


class FastLoader:
    """Small batch loader. It slices big tensors directly, which is much faster
    than building a PyTorch Dataset that returns one row at a time."""

    def __init__(self, X, y, batch_size=2048, shuffle=False, fraud_weight=None, device="cpu"):
        self.X = torch.as_tensor(X, dtype=torch.float32, device=device)
        self.y = torch.as_tensor(y, dtype=torch.float32, device=device)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.fraud_weight = fraud_weight
        self.dataset = self.y        # the trainer only needs len(loader.dataset)
        self.is_training = shuffle or fraud_weight is not None

    def __len__(self):
        return (len(self.y) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        n = len(self.y)
        if self.fraud_weight is not None:
            # oversampling: draw n rows, fraud rows are `fraud_weight` times more likely to be picked
            w = torch.ones_like(self.y)
            w[self.y == 1] = float(self.fraud_weight)
            idx = torch.multinomial(w, n, replacement=True)
        elif self.shuffle:
            idx = torch.randperm(n, device=self.y.device)
        else:
            idx = torch.arange(n, device=self.y.device)

        for i in range(0, n, self.batch_size):
            batch = idx[i:i + self.batch_size]
            if self.is_training and len(batch) < 2:     # BatchNorm cannot train on a single row
                continue
            yield self.X[batch], self.y[batch]


def make_loader(X, y, batch_size=2048, shuffle=False, fraud_weight=None, device="cpu"):
    return FastLoader(X, y, batch_size, shuffle, fraud_weight, device)
