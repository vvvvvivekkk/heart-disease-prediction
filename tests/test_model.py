"""Tests for the HeartMLP model."""
import torch

from src.model import HeartMLP


def test_forward_output_shape():
    model = HeartMLP(input_dim=27)
    model.eval()  # BatchNorm needs eval mode for a single-sample forward
    x = torch.randn(8, 27)
    out = model(x)
    assert out.shape == (8,)


def test_predict_proba_in_unit_interval():
    model = HeartMLP(input_dim=27)
    x = torch.randn(16, 27)
    p = model.predict_proba(x)
    assert p.shape == (16,)
    assert float(p.min()) >= 0.0 and float(p.max()) <= 1.0


def test_configurable_depth():
    model = HeartMLP(input_dim=10, hidden_dims=(32, 16, 8), dropout=0.2)
    n_linear = sum(1 for m in model.modules() if isinstance(m, torch.nn.Linear))
    assert n_linear == 4  # 3 hidden + 1 output


def test_has_trainable_parameters():
    model = HeartMLP(input_dim=27)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) > 0
