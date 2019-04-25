import torch
import torch.nn as nn


def hinge_loss(score_full: torch.Tensor, score_crop: torch.Tensor, g=1.) -> torch.Tensor:
    zeros = torch.zeros_like(score_crop)
    g = torch.tensor(g, device=zeros.device)
    return torch.mean(torch.max(zeros, g + score_crop - score_full))


def ranknet_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return nn.BCEWithLogitsLoss()(x, y)
