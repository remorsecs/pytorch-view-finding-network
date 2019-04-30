from typing import Type, TypeVar

import torch
import torch.nn as nn
from vfn.networks.backbones import Backbone


class ViewFindingNet(nn.Module):

    def __init__(self, backbone: Backbone):
        super().__init__()
        self.backbone = backbone
        self.fc1 = nn.Sequential(
            nn.Linear(self.backbone.output_dim(), 1000),
            nn.ReLU(True),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(1000, 1),
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                m.bias.data.zero_()

    def forward(self, image):
        x = self.backbone(image)    # type: torch.Tensor
        x = x.view((x.size(0), -1))
        x = self.fc1(x)
        score = self.fc2(x)
        return score

