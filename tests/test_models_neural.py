"""Tests verifying PyTorch Neural Network and GNN model architectures."""

import pytest

_no_torch = pytest.mark.skipif(
    not __import__("importlib").util.find_spec("torch"),
    reason="torch not installed",
)


@_no_torch
def test_res_block_output_shape():
    import torch
    from models.mlp import ResBlock
    block = ResBlock(dim=64, dropout=0.0)
    x = torch.randn(8, 64)
    out = block(x)
    assert out.shape == (8, 64)


@_no_torch
def test_fraud_mlp_output_shape():
    import torch
    from models.mlp import FraudMLP
    mlp = FraudMLP(n_features=50, hidden=64, n_blocks=2, dropout=0.0)
    x = torch.randn(16, 50)
    out = mlp(x)
    assert out.shape == (16,)


@_no_torch
def test_fraud_mlp_zero_blocks_is_plain_mlp():
    import torch
    from models.mlp import FraudMLP
    mlp = FraudMLP(n_features=20, hidden=32, n_blocks=0, dropout=0.0)
    x = torch.randn(4, 20)
    out = mlp(x)
    assert out.shape == (4,)
