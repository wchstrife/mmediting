import torch
import torch.nn as nn
from mmedit.models.builder import build_component
from mmedit.models.registry import BACKBONES

@BACKBONES.register_module()
class FBAEncoderDecoder(nn.Module):

    def __init__(self, encoder, decoder):
        super(FBAEncoderDecoder, self).__init__()
        self.encoder = build_component(encoder)
        self.decoder = build_component(decoder)

    def forward(self, image, two_chan_trimap, image_n, trimap_transformed):
        resnet_input = torch.cat((image_n, trimap_transformed, two_chan_trimap), 1) # 3+6+2
        conv_out, indices = self.encoder(resnet_input, return_feature_maps=True)
        return self.decoder(conv_out, image, indices, two_chan_trimap)       

