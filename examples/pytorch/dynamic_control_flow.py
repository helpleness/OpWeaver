"""Intentionally unsupported by symbolic FX; previous valid graph must survive."""

from torch import nn


class DynamicModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(8, 8)

    def forward(self, x):
        if x.sum() > 0:
            return self.layer(x)
        return x
