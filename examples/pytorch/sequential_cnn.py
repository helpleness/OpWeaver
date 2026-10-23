"""Capture with example shape [2, 3, 8, 8]; no external data required."""

import torch
from torch import nn


class ConvolutionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3),
            nn.BatchNorm2d(4),
            nn.GELU(),
        )

    def forward(self, image):
        features = self.features(image)
        return torch.cat((features, features), dim=1)
