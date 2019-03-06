import torch
import torch.nn as nn


def svm_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # Ref: https://pytorch.org/docs/stable/nn.html#marginrankingloss
    criterion = nn.MarginRankingLoss(margin=1)
    return criterion(x, y, torch.ones_like(x))


def ranknet_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return nn.BCEWithLogitsLoss()(x, y)
