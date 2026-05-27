from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


def build_resnet18(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    if pretrained:
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    else:
        m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(reduction="none")

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = self.ce(logits, target)
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss

