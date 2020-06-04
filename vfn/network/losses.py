import torch
import torch.nn as nn


def hinge_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # Ref: https://pytorch.org/docs/stable/nn.html#marginrankingloss
    criterion = nn.MarginRankingLoss(margin=1)
    return criterion(x, y, torch.ones_like(x))


def hinge_loss_v2(score_full: torch.Tensor, score_crop: torch.Tensor, g=1.) -> torch.Tensor:
    zeros = torch.zeros_like(score_crop)
    g = torch.tensor(g, device=zeros.device)
    return torch.mean(torch.max(zeros, g + score_crop - score_full))


def ranknet_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return nn.BCEWithLogitsLoss()(x, y)
