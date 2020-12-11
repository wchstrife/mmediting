import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, kaiming_init, normal_init

from mmedit.models.common import DepthwiseSeparableConvModule
from mmedit.models.registry import COMPONENTS


class IndexedUpsample(nn.Module):
    """Indexed upsample module.

    Args:
        in_channels (int): Input channels.
        out_channels (int): Output channels.
        kernel_size (int, optional): Kernel size of the convolution layer.
            Defaults to 5.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Defaults to dict(type='BN').
        conv_module (ConvModule | DepthwiseSeparableConvModule, optional):
            Conv module. Defaults to ConvModule.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=5,
                 norm_cfg=dict(type='BN'),
                 conv_module=ConvModule):
        super(IndexedUpsample, self).__init__()

        self.conv = conv_module(
            in_channels,
            out_channels,
            kernel_size,
            padding=(kernel_size - 1) // 2,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='ReLU6'))

        self.init_weights()

    def init_weights(self):
        """Init weights for the module.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x, shortcut, dec_idx_feat=None):
        """Forward function.

        Args:
            x (Tensor): Input feature map with shape (N, C, H, W).
            shortcut (Tensor): The shorcut connection with shape
                (N, C, H', W').
            dec_idx_feat (Tensor, optional): The decode index feature map with
                shape (N, C, H', W'). Defaults to None.

        Returns:
            Tensor: Output tensor with shape (N, C, H', W').
        """
        if dec_idx_feat is not None:
            assert shortcut.dim() == 4, (
                'shortcut must be tensor with 4 dimensions')
            x = dec_idx_feat * F.interpolate(x, size=shortcut.shape[2:])
        out = torch.cat((x, shortcut), dim=1)
        return self.conv(out)


@COMPONENTS.register_module()
class IndexNetDecoderFG(nn.Module):

    def __init__(self,
                 in_channels,
                 kernel_size=5,
                 norm_cfg=dict(type='BN'),
                 separable_conv=False):
        # TODO: remove in_channels argument
        super(IndexNetDecoderFG, self).__init__()

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

        self.pred = nn.Sequential(
            conv_module(
                32,
                7,
                kernel_size,
                padding=(kernel_size - 1) // 2,
                norm_cfg=norm_cfg,
                act_cfg=dict(type='ReLU6')),
            nn.Conv2d(
                7, 7, kernel_size, padding=(kernel_size - 1) // 2, bias=False))

    def init_weights(self):
        """Init weights for the module.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                std = math.sqrt(2. / (m.out_channels * m.kernel_size[0]**2))
                normal_init(m, mean=0, std=std)

    def forward(self, inputs, ori_img):
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

        output = self.pred(out)

        alpha = torch.clamp(output[:, 0][:, None], 0, 1)    # [B, 1, H, W]
        F = torch.sigmoid(output[:, 1:4])
        B = torch.sigmoid(output[:, 4:7])

        # FBA Fusion
        #alpha, F, B = fba_fusion(alpha, ori_img, F, B)

        output = torch.cat((alpha, F, B), 1)

        return output
        
def fba_fusion(alpha, img, F, B):
    F = ((alpha * img + (1 - alpha**2) * F - alpha * (1 - alpha) * B))
    B = ((1 - alpha) * img + (2 * alpha - alpha**2) * B - alpha * (1 - alpha) * F)

    F = torch.clamp(F, 0, 1)
    B = torch.clamp(B, 0, 1)
    la = 0.1
    alpha = (alpha * la + torch.sum((img - B) * (F - B), 1, keepdim=True)) / (torch.sum((F - B) * (F - B), 1, keepdim=True) + la)
    alpha = torch.clamp(alpha, 0, 1)
    return alpha, F, B
