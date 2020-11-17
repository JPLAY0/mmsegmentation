from timm import create_model
from torch import nn

from ..builder import BACKBONES


@BACKBONES.register_module()
class BaseNet(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.net = create_model(name, pretrained=True, features_only=True, out_indices=(1, 2, 3, 4))

    def forward(self, x):
        return self.net(x)

    def init_weights(self, pretrained=None):
        pass
