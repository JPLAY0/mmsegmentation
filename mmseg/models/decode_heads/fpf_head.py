import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from .decode_head import BaseDecodeHead
from .psp_head import PPM
from ..builder import HEADS


@HEADS.register_module()
class FPFHead(BaseDecodeHead):

    def __init__(self, pool_scales=(1, 2, 3, 6), **kwargs):
        super(FPFHead, self).__init__(**kwargs)

        self.pool_scales = pool_scales

        self.psp_modules = PPM(
            self.pool_scales,
            self.in_channels[-1],
            self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            align_corners=self.align_corners)
        self.bottleneck = ConvModule(
            self.in_channels[-1] + len(pool_scales) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.convs = nn.ModuleList()
        self.num_convs = len(self.in_channels) - 1
        for i in range(self.num_convs):
            self.convs.append(ConvModule(self.in_channels[i], self.channels, 1, norm_cfg=self.norm_cfg, inplace=False))
            self.convs.append(ConvModule(self.channels, self.channels, 3, padding=1, norm_cfg=self.norm_cfg))
        self.up = nn.Upsample(mode='bilinear', align_corners=False)

    def forward(self, inputs):
        """Forward function."""
        x = self._transform_inputs(inputs)

        psp_outs = [x[-1]]
        psp_outs.extend(self.psp_modules(x[-1]))
        psp_outs = torch.cat(psp_outs, dim=1)
        x[-1] = self.bottleneck(psp_outs)

        for i in reversed(range(self.num_convs)):
            x[i] = self.convs[i * 2](x[i])
            self.up.size = x[i].shape[2:]
            x[i] = x[i] + self.up(x[i + 1])
            x[i] = self.convs[i * 2 + 1](x[i])
        return self.cls_seg(x[0])
