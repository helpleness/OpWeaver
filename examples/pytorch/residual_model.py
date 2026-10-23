"""Offline P0-A example: construction, normalization, activation and residual dataflow."""

from torch import nn


def describe_model():
    """Unrelated hand-written code that graph edits must preserve."""
    return "Residual classifier for OpWeaver's code-first workbench"


class ResidualModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(8, 8)
        self.norm = nn.LayerNorm(8)
        self.activation = nn.ReLU()
        self.head = nn.Linear(8, 3)

    def forward(self, x):
        hidden = self.activation(self.norm(self.projection(x)))
        return self.head(hidden + x)
