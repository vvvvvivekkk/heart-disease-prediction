"""The deep-learning model: a small fully-connected neural network (MLP).

The dataset is tabular and modest in size (~300 rows), so a compact network
with batch normalisation and dropout is a sensible, well-regularised choice.
The network outputs a single logit; pair it with ``BCEWithLogitsLoss``.
"""

from __future__ import annotations

import torch
from torch import nn


class HeartMLP(nn.Module):
    """A configurable multi-layer perceptron for binary classification."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...] = (64, 32),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, 1))  # single logit
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return P(heart disease) in [0, 1]."""
        self.eval()
        return torch.sigmoid(self.forward(x))
