from __future__ import annotations

import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class HospitalProbe(nn.Module):
    """ResNet-18 classifier for predicting CAMELYON17 hospital/center."""

    def __init__(self, num_centers: int = 5, pretrained: bool = True) -> None:
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.encoder = resnet18(weights=weights)
        in_features = self.encoder.fc.in_features
        self.encoder.fc = nn.Linear(in_features, num_centers)

    def forward(self, x):
        return self.encoder(x)
