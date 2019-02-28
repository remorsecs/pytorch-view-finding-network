import pytest
import numpy as np
import torch

from vfn.networks.losses import svm_loss


def np_build_loss_matrix(batch_size):
    loss_matrix = np.zeros(shape=(batch_size, batch_size * 2), dtype=np.float32)
    for k in range(batch_size):
        loss_matrix[k, k] = 1
        loss_matrix[k, k+batch_size] = -1
    return loss_matrix


def np_svm_loss(feature_vec, loss_matrix):
    # q = score(feature_vec)
    q = np.random.uniform()
    p = loss_matrix * q
    zero = np.zeros([1], dtype=np.float32)
    p_hinge = np.maximum(zero, 1+p)
    L = np.mean(p_hinge)
    return L, p


@pytest.mark.skip(reason='')
def test_np_build_loss_matrix():
    pass


@pytest.mark.skip(reason='')
def test_np_svm_loss():
    loss_matrix = np_build_loss_matrix(5)
    L, p = np_svm_loss(None, loss_matrix)


def test_svm_loss():
    x = torch.randn((5, 1), requires_grad=True)
    y = torch.randn((5, 1), requires_grad=True)

    ground_truth = torch.max(torch.zeros_like(x), y - x + 1)
    ground_truth = torch.mean(ground_truth)

    loss = svm_loss(x, y)

    # test values are correct
    assert torch.all(torch.lt(torch.abs(ground_truth - loss), 1e-12))

    # test it can run backpropagation
    assert loss.backward() is None

