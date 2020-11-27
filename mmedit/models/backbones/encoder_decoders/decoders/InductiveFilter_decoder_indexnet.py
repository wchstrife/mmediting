import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, kaiming_init, normal_init
from mmedit.models.common import DepthwiseSeparableConvModule
from mmedit.models.registry import COMPONENTS
from .indexnet_decoder import IndexedUpsample

@COMPONENTS.register_module()
class IGFIndexNetDecoder(nn.Module):

    def __init__(self,
                 in_channels,
                 kernel_size=5,
                 norm_cfg=dict(type='BN'),
                 separable_conv=False):
        # TODO: remove in_channels argument
        super(IGFIndexNetDecoder, self).__init__()

        if separable_conv:
            conv_module = DepthwiseSeparableConvModule
        else:
            conv_module = ConvModule

        blocks_in_channels = [
            in_channels * 2, 96 * 2, 64 * 2, 32 * 2, 24 * 2, 16 * 2, 32 * 2
        ]
        blocks_out_channels = [96, 64, 32, 24, 16, 32, 32]

        self.decoder_layers = nn.ModuleList()
        for in_channels, out_channels in zip(blocks_in_channels,
                                             blocks_out_channels):
            self.decoder_layers.append(
                IndexedUpsample(in_channels, out_channels, kernel_size,
                                norm_cfg, conv_module))

        self.predA = nn.Sequential(
            conv_module(
                32,
                3,
                kernel_size,
                padding=(kernel_size - 1) // 2,
                norm_cfg=norm_cfg,
                act_cfg=dict(type='ReLU6')))
        self.predB = nn.Sequential(
            conv_module(
                32,
                1,
                kernel_size,
                padding=(kernel_size - 1) // 2,
                norm_cfg=norm_cfg,
                act_cfg=dict(type='ReLU6')),
            )
        '''
        self.predC = conv_module(32, 32, 3, padding=1,  norm_cfg=norm_cfg, act_cfg=dict(type='ReLU6'))
        self.predc = nn.Sequential(
            conv_module(
                33,
                32,
                kernel_size,
                padding=(kernel_size - 1) // 2,
                norm_cfg=norm_cfg,
                act_cfg=dict(type='ReLU6')),
            conv_module(
                32,
                32,
                kernel_size,
                padding=(kernel_size - 1) // 2,
                norm_cfg=norm_cfg,
                act_cfg=dict(type='ReLU6')),
            conv_module(
                32,
                32,
                kernel_size,
                padding=(kernel_size - 1) // 2,
                norm_cfg=norm_cfg,
                act_cfg=dict(type='ReLU6')),
            conv_module(
                32,
                3,
                1,
                padding=0,
                norm_cfg=None,
                act_cfg=None),
            )
        '''

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
        for decode_layer, shortcut, dec_idx_feat in zip(*group):
            out = decode_layer(out, shortcut, dec_idx_feat)

        outA = self.predA(out)
        outB = self.predB(out)
        #trimap = self.predC(out)
        alpha = torch.sum(outA * inputs['ori_input'][:,:-1,:,:], dim=1, keepdim=True) + outB
        #trimap = torch.cat((alpha, trimap), 1)
        #trimap = self.predc(trimap)
        return alpha#, trimap
