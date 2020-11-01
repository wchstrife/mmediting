import torch
import torch.nn as nn

import mmedit.models.backbones.encoder_decoders.encoders.fba_resnet_bn as resnet_bn
import mmedit.models.backbones.encoder_decoders.encoders.fba_resnet_GN_WS as resnet_GN_WS

from mmedit.models.registry import COMPONENTS
from mmedit.utils.logger import get_root_logger
from mmcv.runner import load_checkpoint

def build_encoder(self, arch='resnet50_GN_WS'):
    if arch == 'resnet50_GN_WS':
        orig_resnet = resnet_GN_WS.__dict__['l_resnet50']()
        net_encoder = ResnetDilated(orig_resnet, dilate_scale=8)
    elif arch == 'resnet50_BN':
        orig_resnet = resnet_bn.__dict__['l_resnet50']()
        net_encoder = ResnetDilatedBN(orig_resnet, dilate_scale=8)

    else:
        raise Exception('Architecture undefined!')

    num_channels = 3 + 6 + 2

    if(num_channels > 3):
        print(f'modifying input layer to accept {num_channels} channels')
        net_encoder_sd = net_encoder.state_dict()
        conv1_weights = net_encoder_sd['conv1.weight']      

        c_out, c_in, h, w = conv1_weights.size()
        conv1_mod = torch.zeros(c_out, num_channels, h, w)
        conv1_mod[:, :3, :, :] = conv1_weights              

        conv1 = net_encoder.conv1
        conv1.in_channels = num_channels
        conv1.weight = torch.nn.Parameter(conv1_mod)

        net_encoder.conv1 = conv1

        net_encoder_sd['conv1.weight'] = conv1_mod

        net_encoder.load_state_dict(net_encoder_sd)
    return net_encoder

@COMPONENTS.register_module()
class FBAEncoder(nn.Module):
    def __init__(self, in_channels, block):
        super(FBAEncoder, self).__init__()
        assert in_channels == 11, (f'in_channels must be 11, but got {in_channels}')

        if block == 'resnet50_GN_WS':
            orig_resnet = resnet_GN_WS.__dict__['l_resnet50']()
            net_encoder = ResnetDilated(orig_resnet, dilate_scale=8)
        elif block == 'resnet50_BN':
            orig_resnet = resnet_bn.__dict__['l_resnet50']()
            net_encoder = ResnetDilatedBN(orig_resnet, dilate_scale=8)
        else:
            raise Exception('Architecture undefined!')

        if(in_channels > 3):
            print(f'modifying input layer to accept {in_channels} channels')
            net_encoder_sd = net_encoder.state_dict()
            conv1_weights = net_encoder_sd['conv1.weight']      

            c_out, c_in, h, w = conv1_weights.size()
            conv1_mod = torch.zeros(c_out, in_channels, h, w)
            conv1_mod[:, :3, :, :] = conv1_weights              

            conv1 = net_encoder.conv1
            conv1.in_channels = in_channels
            conv1.weight = torch.nn.Parameter(conv1_mod)

            net_encoder.conv1 = conv1

            net_encoder_sd['conv1.weight'] = conv1_mod

            net_encoder.load_state_dict(net_encoder_sd)

            self.encoder = net_encoder
   
    
    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            self.encoder.conv1.weight.data[:, 3:, :, :] = 0
        
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            pass
        else:
            raise TypeError(f'"pretrained" must be a str or None.' f'But received {type(pretrained)}')

    def forward(self, x, return_feature_maps=False):
        return self.encoder(x, return_feature_maps)

class ResnetDilatedBN(nn.Module):
    def __init__(self, orig_resnet, dilate_scale=8):
        super(ResnetDilatedBN, self).__init__()
        from functools import partial

        if dilate_scale == 8:
            orig_resnet.layer3.apply(
                partial(self._nostride_dilate, dilate=2))
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=4))
        elif dilate_scale == 16:
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=2))

        # take pretrained resnet, except AvgPool and FC
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu1
        self.conv2 = orig_resnet.conv2
        self.bn2 = orig_resnet.bn2
        self.relu2 = orig_resnet.relu2
        self.conv3 = orig_resnet.conv3
        self.bn3 = orig_resnet.bn3
        self.relu3 = orig_resnet.relu3
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate // 2, dilate // 2)
                    m.padding = (dilate // 2, dilate // 2)
            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x, return_feature_maps=False):
        conv_out = [x]
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        conv_out.append(x)
        x, indices = self.maxpool(x)
        x = self.layer1(x)
        conv_out.append(x)
        x = self.layer2(x)
        conv_out.append(x)
        x = self.layer3(x)
        conv_out.append(x)
        x = self.layer4(x)
        conv_out.append(x)

        if return_feature_maps:
            return conv_out, indices
        return [x]


class Resnet(nn.Module):
    def __init__(self, orig_resnet):
        super(Resnet, self).__init__()

        # take pretrained resnet, except AvgPool and FC
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu1
        self.conv2 = orig_resnet.conv2
        self.bn2 = orig_resnet.bn2
        self.relu2 = orig_resnet.relu2
        self.conv3 = orig_resnet.conv3
        self.bn3 = orig_resnet.bn3
        self.relu3 = orig_resnet.relu3
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

    def forward(self, x, return_feature_maps=False):
        conv_out = []

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        conv_out.append(x)
        x, indices = self.maxpool(x)

        x = self.layer1(x)
        conv_out.append(x)
        x = self.layer2(x)
        conv_out.append(x)
        x = self.layer3(x)
        conv_out.append(x)
        x = self.layer4(x)
        conv_out.append(x)

        if return_feature_maps:
            return conv_out
        return [x]


class ResnetDilated(nn.Module):
    def __init__(self, orig_resnet, dilate_scale=8):
        super(ResnetDilated, self).__init__()
        from functools import partial

        if dilate_scale == 8:
            orig_resnet.layer3.apply(
                partial(self._nostride_dilate, dilate=2))
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=4))
        elif dilate_scale == 16:
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=2))

        # take pretrained resnet, except AvgPool and FC
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu = orig_resnet.relu
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate // 2, dilate // 2)
                    m.padding = (dilate // 2, dilate // 2)
            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x, return_feature_maps=False):
        conv_out = [x]
        x = self.relu(self.bn1(self.conv1(x)))
        conv_out.append(x)
        x, indices = self.maxpool(x)
        x = self.layer1(x)
        conv_out.append(x)
        x = self.layer2(x)
        conv_out.append(x)
        x = self.layer3(x)
        conv_out.append(x)
        x = self.layer4(x)
        conv_out.append(x)

        if return_feature_maps:
            return conv_out, indices
        return [x]

        


