import torch
import torch.nn as nn

from ..builder import build_loss
from ..registry import MODELS
from .base_mattor import BaseMattor


@MODELS.register_module()
class FBA(BaseMattor):


    def __init__(self, backbone, train_cfg=None, test_cfg=None, pretrained=None, loss_alpha=None, loss_comp=None):
        super(FBA, self).__init__(backbone, None, train_cfg, test_cfg, pretrained)

        # TODO: 添加其他Loss
        self.loss_alpha = (build_loss(loss_alpha) if loss_alpha is not None else None)
        self.loss_comp = (build_loss(loss_comp) if loss_comp is not None else None)

    def forward_dummy(self, inputs):
        return self.backbone(inputs)

    # TODO: 添加训练代码
    def forward_train(self, parameter_list):
        pass

    # TODO： 添加参数注释
    def forward_test(self, image, two_chan_trimap, image_n, trimap_transformed, meta, save_image=False, save_path=None, iteration=None):

        resnet_input = torch.cat((image_n, trimap_transformed, two_chan_trimap), 1)
        result = self.backbone(resnet_input, image)
        
        # TODO: 分离Decoder的输出
        pred_alpha = result.cpu().numpy().squeeze()
        pred_alpha = self.restore_shape(pred_alpha, meta)
        eval_result = self.evaluate(pred_alpha, meta)

        if save_image:
            self.save_image(pred_alpha, meta, save_path, iteration)

        return {'pred_alpha': pred_alpha, 'eval_result': eval_result}
        
