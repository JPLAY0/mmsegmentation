from torch import nn

from .decode_head import BaseDecodeHead
from ..builder import HEADS


@HEADS.register_module()
class FASTHead(BaseDecodeHead):

    def __init__(self,
                 **kwargs):
        super(FASTHead, self).__init__(**kwargs)
        self.convs = nn.ModuleList()
        self.num_convs = len(self.in_channels) - 1
        for i in range(self.num_convs):
            self.convs.append(nn.ConvTranspose2d(self.in_channels[i + 1], self.in_channels[i], 2, 2, bias=False))

    def forward(self, inputs):
        """Forward function."""
        x = self._transform_inputs(inputs)
        for i in reversed(range(self.num_convs)):
            x[i] += self.convs[i](x[i + 1])
        output = self.cls_seg(x[0])
        return output
