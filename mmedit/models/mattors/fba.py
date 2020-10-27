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

    def forward(self,
                ori_merged,
                trimap,
                merged,
                trimap_transformed,
                meta,
                alpha=None,
                test_mode=False,
                **kwargs):
        if not test_mode:
            return self.forward_train(merged, trimap, meta, alpha, **kwargs)
        else:
            return self.forward_test(ori_merged, trimap, merged, trimap_transformed, meta, **kwargs)

    # TODO: 添加训练代码
    def forward_train(self, parameter_list):
        pass

    # TODO： 添加参数注释
    def forward_test(self, ori_merged, trimap, merged, trimap_transformed, meta, save_image=False, save_path=None, iteration=None):

        # resnet_input = torch.cat((image_n, trimap_transformed, two_chan_trimap), 1)
        # print(resnet_input.shape)

        result = self.backbone(ori_merged, trimap, merged, trimap_transformed)
        result = result.cpu().numpy().squeeze()
        trimap = trimap.cpu().numpy().squeeze()
        # print(result.shape)
        
        pred_alpha = result[0, :, :]
        fg = result[1:4, :, :]
        bg = result[4:7, :, :]

        pred_alpha[trimap[0, :, :] == 1] = 0
        pred_alpha[trimap[1, :, :] == 1] = 1 

        pred_alpha = self.restore_shape(pred_alpha, meta)
        eval_result = self.evaluate(pred_alpha, meta)
        

        if save_image:
            self.save_image(pred_alpha, meta, save_path, iteration)

        return {'pred_alpha': pred_alpha, 'eval_result': eval_result, 'fg': fg, 'bg': bg}

        
