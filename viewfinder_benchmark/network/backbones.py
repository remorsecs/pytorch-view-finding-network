# CNN backbone for vfn
import torch
import torch.nn as nn
import torchvision.models as models


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, *input):
        pass

    def input_dim(self):
        pass

    def output_dim(self):
        pass


# Example:
#   backbone = backbones.AlexNet(pretrained)
#   vfn = ViewFindingNet(backbone)
#   ...
class AlexNet(Backbone):

    def __init__(self, pretrained=False):
        super().__init__()
        model = models.alexnet(pretrained)
        self.features = model.features
        self.avgpool = model.avgpool

    def forward(self, x: torch.Tensor):
        x = self.features(x)
        x = self.avgpool(x)
        return x

    def input_dim(self):
        return 227

    def output_dim(self):
        # x = torch.randn((1, 3, self.input_dim(), self.input_dim()))
        # y = self.forward(x)
        # y = y.view((-1))
        # Output: 256 * 6 * 6 = 9216
        return 9216


class VGG(Backbone):
    def __init__(self, pretrained=False):
        super().__init__()
        model = models.vgg16(pretrained)
        self.features = model.features
        self.avgpool = model.avgpool

    def forward(self, x: torch.Tensor):
        x = self.features(x)
        x = self.avgpool(x)
        return x

    def input_dim(self):
        return 224

    def output_dim(self):
        # x = torch.randn((1, 3, self.input_dim(), self.input_dim()))
        # y = self.forward(x)
        # y = y.view((-1))
        # print(y.size())
        # Output: 512 * 7 * 7 = 25088
        return 25088
