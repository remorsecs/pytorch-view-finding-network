from functools import partial

import torch
import torch.optim as optim
import yaml

from torch.utils.data import DataLoader
from torchvision import transforms

import viewfinder_benchmark.network.backbones as backbones
import viewfinder_benchmark.network.losses as losses
import viewfinder_benchmark.network.models as models

from viewfinder_benchmark.data.FCDB import FCDB
from viewfinder_benchmark.data.ICDB import ICDB
from viewfinder_benchmark.data.dataset import ImagePairDataset


class ConfigParser:

    def __init__(self, config_file):
        self._load(config_file)
        self._init_name()
        self.input_dim = 0

    def _init_name(self):
        # All `*_name` are disposable because the `pop()` operation will modify the origin `configs`.
        self.backbone_name = self.configs['model']['backbone'].pop('name')
        self.optimizer_name = self.configs['train']['optimizer'].pop('name')
        self.loss_name = self.configs['train']['loss'].pop('name')
        self.dataset_name = self.configs['train']['dataset'].pop('name')

    def _load(self, config_file):
        with open(config_file, 'r') as f:
            self.configs = yaml.load(f, Loader=yaml.FullLoader)

    def get_model_name(self):
        return self.backbone_name

    def parse_model(self):
        backbone_model = None
        if self.backbone_name == 'AlexNet':
            backbone_model = backbones.AlexNet
        elif self.backbone_name == 'VGG':
            backbone_model = backbones.VGG

        backbone = backbone_model(**self.configs['model']['backbone'])
        model = models.ViewFindingNet(backbone=backbone)
        self.input_dim = backbone.input_dim()
        return model

    def parse_optimizer(self):
        optimizer_fn = None

        if self.optimizer_name == 'SGD':
            optimizer_fn = optim.SGD
        elif self.optimizer_name == 'Adam':
            optimizer_fn = optim.Adam

        optimizer = partial(optimizer_fn, **self.configs['train']['optimizer'])
        return optimizer

    def parse_loss_function(self):
        loss_fn = None

        if self.loss_name == 'hinge':
            loss_fn = losses.hinge_loss
        elif self.loss_name == 'ranknet':
            loss_fn = losses.ranknet_loss

        return loss_fn

    def parse_dataloader(self):

        # build data augmentation transforms
        data_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.input_dim, self.input_dim)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.01, contrast=0.05),
            transforms.ToTensor(),
        ])

        train_dataset = ImagePairDataset(self.configs['train']['dataset']['gulpio_dir'],
                                         data_transform)
        val_dataset = ImagePairDataset(self.configs['validation']['dataset']['gulpio_dir'],
                                       data_transform)

        print('train_size:', len(train_dataset))
        print('val_size:', len(val_dataset))

        data_loaders = dict(
            train=DataLoader(train_dataset, num_workers=8, **self.configs['train']['dataloader']),
            val=DataLoader(val_dataset, num_workers=8, **self.configs['validation']['dataloader']),
        )
        return data_loaders

    def parse_device(self):
        return torch.device(self.configs['device'])

    def parse_FCDB(self):
        return FCDB(**self.configs['evaluate']['FCDB'])

    def parse_ICDB(self, subset_selector):
        return ICDB(subset=subset_selector, **self.configs['evaluate']['ICDB'])
