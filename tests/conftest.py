import pytest
import torch
from torchvision.transforms import transforms as T

from tests.test_evaluation import SimpleImageCropper
from viewfinder_benchmark.data.evaluation import ImageCropperEvaluator
from viewfinder_benchmark.data.FCDB import FCDB


def pytest_addoption(parser):
    parser.addoption('--fcdb', type=str, default='')


@pytest.fixture(scope='module')
def root_fcdb(request):
    return request.config.getoption('--fcdb')


@pytest.fixture(scope='module')
def evaluator_simple_net_on_FCDB(root_fcdb):
    model = SimpleImageCropper()
    dataset = FCDB(root_fcdb, download=False)
    device = torch.device('cuda:0')
    transforms = T.Compose([
        T.ToPILImage(),
        T.Resize((16, 16)),
        T.ToTensor(),
    ])
    return ImageCropperEvaluator(model, dataset, device, transforms)
