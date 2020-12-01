import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, kaiming_init, normal_init
from mmedit.models.common import DepthwiseSeparableConvModule
from mmedit.models.registry import COMPONENTS
from .indexnet_decoder import IndexedUpsample
@COMPONENTS.register_module()
class IGFIndexNetDecoderTrimap(nn.Module):

    def __init__(self,
                 in_channels,
                 kernel_size=5,
                 norm_cfg=dict(type='BN'),
                 separable_conv=False,
                 select_layer=(0, 1, 2, 3, 4, 5, 6)):
        # TODO: remove in_channels argument
        super(IGFIndexNetDecoderTrimap, self).__init__()
        self.select_layer=set(select_layer)
        if separable_conv:
            conv_module = DepthwiseSeparableConvModule
        else:
            conv_module = ConvModule

        blocks_in_channels = [
            in_channels * 2, 96 * 2, 64 * 2, 32 * 2, 24 * 2, 16 * 2, 32 * 2
        ]
        blocks_out_channels = [96, 64, 32, 24, 16, 32, 32]

        self.decoder_layers = nn.ModuleList()
        i = 7
        last_channel = 96
        for in_channels, out_channels in zip(blocks_in_channels,
                                             blocks_out_channels):
            i -= 1
            if i not in self.select_layer:
                break
            self.decoder_layers.append(
                IndexedUpsample(in_channels, out_channels, kernel_size,
                                norm_cfg, conv_module))
            last_channel = out_channels

        self.pred = nn.Sequential(
            conv_module(
                last_channel,
                last_channel,
                kernel_size,
                padding=(kernel_size - 1) // 2,
                norm_cfg=norm_cfg,
                act_cfg=dict(type='ReLU6')),
            nn.Conv2d(
                last_channel, 3, 1, padding=0, bias=False))

    def init_weights(self):
        """Init weights for the module.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                std = math.sqrt(2. / (m.out_channels * m.kernel_size[0]**2))
                normal_init(m, mean=0, std=std)

    def forward(self, inputs):
        """Forward fucntion.

        Args:
            inputs (dict): Output dict of IndexNetEncoder.

        Returns:
            Tensor: Predicted alpha matte of the current batch.
        """
        shortcuts = reversed(inputs['shortcuts'])
        dec_idx_feat_list = reversed(inputs['dec_idx_feat_list'])
        out = inputs['out']

        group = (self.decoder_layers, shortcuts, dec_idx_feat_list)
        i = 7
        for decode_layer, shortcut, dec_idx_feat in zip(*group):
            i -= 1
            if i not in self.select_layer:
                break
            out = decode_layer(out, shortcut, dec_idx_feat)
        trimap = self.pred(out)
        return trimap
