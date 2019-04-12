import pytest
import numpy as np
import torch

from vfn.networks.losses import hinge_loss


def test_svm_loss():
    x = torch.randn((5, 1), requires_grad=True)
    y = torch.randn((5, 1), requires_grad=True)

    ground_truth = torch.max(torch.zeros_like(x), y - x + 1)
    ground_truth = torch.mean(ground_truth)

    loss = hinge_loss(x, y)

    # test values are correct
    assert torch.all(torch.lt(torch.abs(ground_truth - loss), 1e-12))

    # test it can run backpropagation
    assert loss.backward() is None

