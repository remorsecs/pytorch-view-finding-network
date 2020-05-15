# To run this test code:
#   $ py.test test_evaluation.py --fcdb=/path/to/FCDB/dataset
from itertools import chain

import pytest
import torch
import torch.nn as nn

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def generate_crops_v1(width, height, device=None):
    if device is None:
        device = torch.device('cpu')

    d_x = list(chain.from_iterable([i * j] * 5 for i in range(10, 0, -2) for j in range(5)))
    d_y = list(chain.from_iterable([i * j for i in range(5)] * 5 for j in range(10, 0, -2)))
    d_len = list(chain.from_iterable([i] * 25 for i in range(5, 10)))

    d_x = torch.tensor(d_x, dtype=torch.float32, device=device) * 0.01
    d_y = torch.tensor(d_y, dtype=torch.float32, device=device) * 0.01
    d_len = torch.tensor(d_len, dtype=torch.float32, device=device) * 0.1

    x = width * d_x
    y = height * d_y
    w = width * d_len
    h = height * d_len

    crops = torch.stack([x, y, w, h]).int().t_().tolist()
    return crops


def generate_crops_v2(width, height, device=None):
    if device is None:
        device = torch.device('cpu')

    d_x = [0, 0, 0, 0, 0, 10, 10, 10, 10, 10, 20, 20, 20, 20, 20, 30, 30, 30, 30, 30, 40, 40, 40, 40, 40,
           0, 0, 0, 0, 0, 8, 8, 8, 8, 8, 16, 16, 16, 16, 16, 24, 24, 24, 24, 24, 32, 32, 32, 32, 32,
           0, 0, 0, 0, 0, 6, 6, 6, 6, 6, 12, 12, 12, 12, 12, 18, 18, 18, 18, 18, 24, 24, 24, 24, 24,
           0, 0, 0, 0, 0, 4, 4, 4, 4, 4, 8, 8, 8, 8, 8, 12, 12, 12, 12, 12, 16, 16, 16, 16, 16,
           0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 8, 8, 8, 8, 8, ]
    d_y = [0, 10, 20, 30, 40, 0, 10, 20, 30, 40, 0, 10, 20, 30, 40, 0, 10, 20, 30, 40, 0, 10, 20, 30, 40,
           0, 8, 16, 24, 32, 0, 8, 16, 24, 32, 0, 8, 16, 24, 32, 0, 8, 16, 24, 32, 0, 8, 16, 24, 32,
           0, 6, 12, 18, 24, 0, 6, 12, 18, 24, 0, 6, 12, 18, 24, 0, 6, 12, 18, 24, 0, 6, 12, 18, 24,
           0, 4, 8, 12, 16, 0, 4, 8, 12, 16, 0, 4, 8, 12, 16, 0, 4, 8, 12, 16, 0, 4, 8, 12, 16,
           0, 2, 4, 6, 8, 0, 2, 4, 6, 8, 0, 2, 4, 6, 8, 0, 2, 4, 6, 8, 0, 2, 4, 6, 8, ]
    d_len = [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
             6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
             7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
             8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
             9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, ]

    d_x = torch.tensor(d_x, dtype=torch.float32, device=device) * 0.01
    d_y = torch.tensor(d_y, dtype=torch.float32, device=device) * 0.01
    d_len = torch.tensor(d_len, dtype=torch.float32, device=device) * 0.1

    x = width * d_x
    y = height * d_y
    w = width * d_len
    h = height * d_len

    crops = torch.stack([x, y, w, h]).int().t_().tolist()
    return crops


def generate_crops_v3(width, height):
    d_x = [0, 0, 0, 0, 0, 10, 10, 10, 10, 10, 20, 20, 20, 20, 20, 30, 30, 30, 30, 30, 40, 40, 40, 40, 40,
           0, 0, 0, 0, 0, 8, 8, 8, 8, 8, 16, 16, 16, 16, 16, 24, 24, 24, 24, 24, 32, 32, 32, 32, 32,
           0, 0, 0, 0, 0, 6, 6, 6, 6, 6, 12, 12, 12, 12, 12, 18, 18, 18, 18, 18, 24, 24, 24, 24, 24,
           0, 0, 0, 0, 0, 4, 4, 4, 4, 4, 8, 8, 8, 8, 8, 12, 12, 12, 12, 12, 16, 16, 16, 16, 16,
           0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 8, 8, 8, 8, 8, ]
    d_y = [0, 10, 20, 30, 40, 0, 10, 20, 30, 40, 0, 10, 20, 30, 40, 0, 10, 20, 30, 40, 0, 10, 20, 30, 40,
           0, 8, 16, 24, 32, 0, 8, 16, 24, 32, 0, 8, 16, 24, 32, 0, 8, 16, 24, 32, 0, 8, 16, 24, 32,
           0, 6, 12, 18, 24, 0, 6, 12, 18, 24, 0, 6, 12, 18, 24, 0, 6, 12, 18, 24, 0, 6, 12, 18, 24,
           0, 4, 8, 12, 16, 0, 4, 8, 12, 16, 0, 4, 8, 12, 16, 0, 4, 8, 12, 16, 0, 4, 8, 12, 16,
           0, 2, 4, 6, 8, 0, 2, 4, 6, 8, 0, 2, 4, 6, 8, 0, 2, 4, 6, 8, 0, 2, 4, 6, 8, ]
    d_len = [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
             6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
             7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
             8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
             9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, ]
    x = list(map(lambda item: int(item * width * 0.01), d_x))
    y = list(map(lambda item: int(item * height * 0.01), d_y))
    w = list(map(lambda item: int(item * width * 0.1), d_len))
    h = list(map(lambda item: int(item * height * 0.1), d_len))

    return list(zip(x, y, w, h))


class SimpleImageCropper(nn.Module):

    def __init__(self):
        super(SimpleImageCropper, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 16 * 3, 1),
        )

    def forward(self, x):
        x = x.view(-1, 16 * 16 * 3)
        return self.fc1(x)


def test_evaluate_iou(evaluator_simple_net_on_FCDB):
    iou_expected = torch.tensor([0.4262])
    iou_actual = evaluator_simple_net_on_FCDB.intersection_over_union
    assert torch.allclose(iou_expected, iou_actual, atol=1e-04)


def test_evaluate_boundary_displacement(evaluator_simple_net_on_FCDB):
    boundary_displacement_expected = torch.tensor([0.1602])
    boundary_displacement_actual = evaluator_simple_net_on_FCDB.boundary_displacement
    assert torch.allclose(boundary_displacement_expected, boundary_displacement_actual, atol=1e-04)


def test_alpha_recall(evaluator_simple_net_on_FCDB):
    alpha_recall_expected = torch.tensor([3.7681])
    alpha_recall_actual = evaluator_simple_net_on_FCDB.alpha_recall
    assert torch.allclose(alpha_recall_expected, alpha_recall_actual, atol=1e-04)


def test_all_metrics(evaluator_simple_net_on_FCDB):
    iou_expected = torch.tensor([0.4262])
    iou_actual = evaluator_simple_net_on_FCDB.intersection_over_union

    boundary_displacement_expected = torch.tensor([0.1602])
    boundary_displacement_actual = evaluator_simple_net_on_FCDB.boundary_displacement

    alpha_recall_expected = torch.tensor([3.7681])
    alpha_recall_actual = evaluator_simple_net_on_FCDB.alpha_recall

    assert torch.allclose(iou_expected, iou_actual, atol=1e-04)
    assert torch.allclose(boundary_displacement_expected, boundary_displacement_actual, atol=1e-04)
    assert torch.allclose(alpha_recall_expected, alpha_recall_actual, atol=1e-04)
