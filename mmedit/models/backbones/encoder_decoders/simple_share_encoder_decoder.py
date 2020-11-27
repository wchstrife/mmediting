import torch
import torch.nn as nn
from mmedit.models.builder import build_component
from mmedit.models.registry import BACKBONES


@BACKBONES.register_module()
class SimpleShareEncoderDecoder(nn.Module):
    """Simple encoder-decoder model from matting.

    Args:
        encoder (dict): Config of the encoder.
        decoder (dict): Config of the decoder.
    """

    def __init__(self, encoder, decoder1, neck, decoder2,
                       train_encoder=True,
                       train_decoder1=True,
                       train_decoder2=True):
        super(SimpleShareEncoderDecoder, self).__init__()

        self.encoder = build_component(encoder)
        self.neck = build_component(neck)
        decoder1['in_channels'] = self.encoder.out_channels
        decoder2['in_channels'] = self.encoder.out_channels
        self.decoder = build_component(decoder1)
        self.decoder2 = build_component(decoder2)
        self.train_encoder = train_encoder
        self.train_decoder1 = train_decoder1
        self.train_decoder2 = train_decoder2

        if not train_encoder:
            self.encoder.eval()
            for param in self.encoder.parameters():
                param.requires_grad = False

        if not train_decoder1:
            self.decoder.eval()
            for param in self.decoder.parameters():
                param.requires_grad = False

        if not train_decoder2:
            self.decoder2.eval()
            for param in self.decoder2.parameters():
                param.requires_grad = False

    def init_weights(self, pretrained=None):
        self.encoder.init_weights(pretrained)
        self.decoder.init_weights()
        self.decoder2.init_weights()

    def forward(self, *args, **kwargs):
        """Forward function.

        Returns:
            Tensor: The output tensor of the decoder.
        """
        if not self.train_encoder:
            self.encoder.eval()

        if not self.train_decoder1:
            self.decoder.eval()

        if not self.train_decoder2:
            self.decoder2.eval()

        out = self.encoder(*args, **kwargs)
        out1 = self.decoder(out) #output trimap

        out = self.neck(out, out1)
        out2 = self.decoder2(out)
        out1 = nn.functional.interpolate(out1, out['ori_input'].shape[-2:], mode='bilinear', align_corners=False)
        return out1, out2
