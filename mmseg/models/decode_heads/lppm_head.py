from torch import nn
from mmcv.cnn import ConvModule

from .decode_head import BaseDecodeHead
from ..builder import HEADS


@HEADS.register_module()
class LPPMHead(BaseDecodeHead):

    def __init__(self, pool_scales=(1, 2, 3, 6), **kwargs):
        super(LPPMHead, self).__init__(**kwargs)
        self.convs = nn.ModuleList()
        self.num_convs = len(self.in_channels) - 1
        self.lppm = LPPM(pool_scales, self.in_channels, self.channels, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg,
                         act_cfg=self.act_cfg, align_corners=self.align_corners)
        for i in range(self.num_convs):
            self.convs.append(ConvModule(self.in_channels[i], self.channels, 1, norm_cfg=self.norm_cfg, inplace=False))
            self.convs.append(ConvModule(self.channels, self.channels, 3, padding=1, norm_cfg=self.norm_cfg))
        self.up = nn.Upsample(mode='bilinear', align_corners=False)

    def forward(self, inputs):
        """Forward function."""
        x = self._transform_inputs(inputs)
        x[-1] = self.lppm(x[-1])
        for i in reversed(range(self.num_convs)):
            x[i] = self.convs[i * 2](x[i])
            self.up.size = x[i].shape[2:]
            x[i] = x[i] + self.up(x[i + 1])
            x[i] = self.convs[i * 2 + 1](x[i])
        return self.cls_seg(x[0])


class LPPM(nn.Module):
    def __init__(self, pool_scales, in_channels, channels, conv_cfg, norm_cfg,
                 act_cfg, align_corners):
        super(LPPM, self).__init__()

        self.pools = nn.ModuleList()
        self.convs = nn.ModuleList()
        self.convs.append(ConvModule(in_channels[-1], channels, 3, padding=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg,
                                     act_cfg=act_cfg))
        for pool_scale in pool_scales:
            self.pools.append(nn.AdaptiveAvgPool2d(pool_scale))
            self.convs.append(ConvModule(in_channels[-1], channels, 3, padding=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg,
                                         act_cfg=act_cfg))
        self.up = nn.Upsample(mode='bilinear', align_corners=align_corners)

    def forward(self, x):
        """Forward function."""
        self.up.size = x.size()[2:]
        copy = x.clone()
        x = self.convs[-1](x)
        for i, pool in enumerate(self.pools):
            x = x + self.up(self.convs[i](pool(copy)))
        return x
