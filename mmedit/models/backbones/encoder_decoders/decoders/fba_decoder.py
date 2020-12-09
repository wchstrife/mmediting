import torch
import torch.nn as nn

# import mmedit.models.common.fba_layer_WS as L
from mmedit.models.common import fba_layer_WS as L

from mmedit.models.registry import COMPONENTS
from mmcv.cnn.utils.weight_init import xavier_init


@COMPONENTS.register_module()
class FBADecoder(nn.Module):

    def __init__(self, batch_norm=False):   # 默认使用GN
        super(FBADecoder, self).__init__()
        pool_scales = (1, 2, 3, 6)
        self.batch_norm = batch_norm

        self.ppm = []

        for scale in pool_scales:
            self.ppm.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                L.Conv2d(2048, 256, kernel_size=1, bias=True),
                norm(256, self.batch_norm),
                nn.LeakyReLU()
            ))
        self.ppm = nn.ModuleList(self.ppm)

        self.conv_up1 = nn.Sequential(
            L.Conv2d(2048 + len(pool_scales) * 256, 256,
                     kernel_size=3, padding=1, bias=True),
            norm(256, self.batch_norm),
            nn.LeakyReLU(),
            L.Conv2d(256, 256, kernel_size=3, padding=1),
            norm(256, self.batch_norm),
            nn.LeakyReLU()
        )

        self.conv_up2 = nn.Sequential(
            L.Conv2d(256 + 256, 256, kernel_size=3, padding=1, bias=True),
            norm(256, self.batch_norm),
            nn.LeakyReLU()            
        )

        if (self.batch_norm):
            d_up3 = 128
        else:
            d_up3 = 64
        
        self.conv_up3 = nn.Sequential(
            L.Conv2d(256 + d_up3, 64, kernel_size=3, padding=1, bias=True),
            norm(64, self.batch_norm),
            nn.LeakyReLU()
        )

        self.unpool = nn.MaxPool2d(2, stride=2)

        self.conv_up4 = nn.Sequential(
            nn.Conv2d(64 + 3 + 3 + 2, 32, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(),
            nn.Conv2d(16, 7, kernel_size=1, padding=0, bias=True)
        )
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m)

    
    def forward(self, conv_out, img, indices, two_chan_trimap):
        conv5 = conv_out[-1]    # encoder最后一层的输出

        input_size = conv5.size()
        ppm_out = [conv5]

        # ppm的输出/decoder的输入
        for pool_scale in self.ppm:
            ppm_out.append(nn.functional.interpolate(
                pool_scale(conv5),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False
            ))
        ppm_out = torch.cat(ppm_out, 1)

        x = self.conv_up1(ppm_out)
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        x = torch.cat((x, conv_out[-4]), 1)

        x = self.conv_up2(x)
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        x = torch.cat((x, conv_out[-5]), 1)

        x = self.conv_up3(x)        
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        x = torch.cat((x, conv_out[-6][:, :3], img, two_chan_trimap), 1)

        output = self.conv_up4(x)

        alpha = torch.clamp(output[:, 0][:, None], 0, 1)    # [B, 1, H, W]
        F = torch.sigmoid(output[:, 1:4])
        B = torch.sigmoid(output[:, 4:7])

        # FBA Fusion
        #alpha, F, B = fba_fusion(alpha, img, F, B)

        output = torch.cat((alpha, F, B), 1)

        return output

def norm(dim, bn=False):
    if (bn is False):
        return nn.GroupNorm(32, dim)
    else:
        return nn.BatchNorm2d(dim)

# TODO: WHY?
def fba_fusion(alpha, img, F, B):
    F = ((alpha * img + (1 - alpha**2) * F - alpha * (1 - alpha) * B))
    B = ((1 - alpha) * img + (2 * alpha - alpha**2) * B - alpha * (1 - alpha) * F)

    F = torch.clamp(F, 0, 1)
    B = torch.clamp(B, 0, 1)
    la = 0.1
    alpha = (alpha * la + torch.sum((img - B) * (F - B), 1, keepdim=True)) / (torch.sum((F - B) * (F - B), 1, keepdim=True) + la)
    alpha = torch.clamp(alpha, 0, 1)
    return alpha, F, B




        




