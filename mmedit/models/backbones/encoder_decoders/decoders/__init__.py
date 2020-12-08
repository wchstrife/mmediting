from .deepfill_decoder import DeepFillDecoder
from .gl_decoder import GLDecoder
from .indexnet_decoder import IndexedUpsample, IndexNetDecoder
from .pconv_decoder import PConvDecoder
from .plain_decoder import PlainDecoder
from .resnet_dec import ResGCADecoder, ResNetDec, ResShortcutDec
from .fba_decoder import FBADecoder
from .InductiveFilter_decoder_indexnet import IGFIndexNetDecoder
from .InductiveFilter_decoder_trimap import IGFIndexNetDecoderTrimap
from .indexnet_decoder_fg import IndexNetDecoderFG
__all__ = [
    'GLDecoder', 'PlainDecoder', 'PConvDecoder', 'ResNetDec', 'ResShortcutDec',
    'DeepFillDecoder', 'IndexedUpsample', 'IndexNetDecoder', 'ResGCADecoder',
    'IGFIndexNetDecoder', 'IGFIndexNetDecoderTrimap', 'FBADecoder', 'IndexNetDecoderFG'
]
