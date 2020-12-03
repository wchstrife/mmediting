import torch
import torch.nn as nn
import cv2
import numpy as np

from ..builder import build_loss
from ..registry import MODELS
from .base_mattor import BaseMattor


@MODELS.register_module()
class FBA(BaseMattor):

    def __init__(self,
                backbone, 
                train_cfg=None, 
                test_cfg=None, 
                pretrained=None, 
                loss_alpha_l1=None,
                loss_alpha_comp=None,
                loss_alpha_grad=None,
                loss_alpha_lap=None,
                loss_f_l1=None,
                loss_b_l1=None,
                loss_fb_excl=None,
                loss_fb_comp=None,
                loss_f_lap=None,
                loss_b_lap=None
                ):
        super(FBA, self).__init__(backbone, None, train_cfg, test_cfg, pretrained)
        
        if all(v is None for v in (loss_alpha_l1,
                                    loss_alpha_comp,
                                    loss_alpha_grad,
                                    loss_alpha_lap,
                                    loss_f_l1,
                                    loss_b_l1,
                                    loss_fb_excl,
                                    loss_fb_comp,
                                    loss_f_lap,
                                    loss_b_lap
                                    )):
            raise ValueError('Please specify one loss for FBA.')

        self.loss_alpha_l1 = (build_loss(loss_alpha_l1) if loss_alpha_l1 is not None else None)
        self.loss_alpha_comp = (build_loss(loss_alpha_comp) if loss_alpha_comp is not None else None)   
        self.loss_alpha_grad = (build_loss(loss_alpha_grad) if loss_alpha_grad is not None else None)
        self.loss_alpha_lap = (build_loss(loss_alpha_lap) if loss_alpha_lap is not None else None)     
        self.loss_f_l1 = (build_loss(loss_f_l1) if loss_f_l1 is not None else None)
        self.loss_b_l1 = (build_loss(loss_b_l1) if loss_b_l1 is not None else None)   
        self.loss_fb_excl = (build_loss(loss_fb_excl) if loss_fb_excl is not None else None)
        self.loss_fb_comp = (build_loss(loss_fb_comp) if loss_fb_comp is not None else None)
        self.loss_f_lap = (build_loss(loss_f_lap) if loss_f_lap is not None else None)
        self.loss_b_lap = (build_loss(loss_b_lap) if loss_b_lap is not None else None) 

         # support fp16
        self.fp16_enabled = False                 
        
        
    def forward_dummy(self, inputs):
        return self.backbone(inputs)

    # def forward(self,
    #             ori_merged,
    #             trimap,
    #             merged,
    #             trimap_transformed,
    #             meta,
    #             alpha=None,
    #             test_mode=False,
    #             **kwargs):
    #     if not test_mode:
    #         return self.forward_train(merged, trimap, meta, alpha, **kwargs)
    #     else:
    #         return self.forward_test(ori_merged, trimap, merged, trimap_transformed, meta, **kwargs)

    # TODO: 添加训练代码
    def forward_train(self, parameter_list):
        pass

    # TODO： 添加参数注释
    def forward_test(self, ori_merged, trimap, merged, trimap_transformed, meta, save_image=False, save_path=None, iteration=None):

        # resnet_input = torch.cat((image_n, trimap_transformed, two_chan_trimap), 1)
        # print(resnet_input.shape)

        result = self.backbone(ori_merged, trimap, merged, trimap_transformed)

        result = self.restore_shape(result, meta) 

        pred_alpha = result[:, :, 0]
        fg = result[:, :, 1:4]
        bg = result[:, :, 4:7]

        ori_trimap = meta[0]['ori_trimap'].squeeze()
        pred_alpha[ori_trimap[:, :, 0] == 1] = 0
        pred_alpha[ori_trimap[:, :, 1] == 1] = 1

        # fg[alpha == 1] = image_np[alpha == 1] # TODO: 需要返回fg和bg时，需要用到merge_np，也就是ori_merge，但是在这里的实现已经经过了插值，改变的话需要重写norm层保留下来原来的ori_merged
        # bg[alpha == 0] = image_np[alpha == 0]

        # result = result.cpu().numpy().squeeze()
        # trimap = trimap.cpu().numpy().squeeze()
        
        eval_result = self.evaluate(pred_alpha, meta)
        

        if save_image:
            self.save_image(pred_alpha, meta, save_path, iteration)

        return {'pred_alpha': pred_alpha, 'eval_result': eval_result}

        
    def restore_shape(self, result, meta):
        """Restore the result to the original shape.

        The shape of the predicted alpha may not be the same as the shape of
        original input image. This function restores the shape of the predicted
        alpha.

        Args:
            pred_alpha (np.ndarray): The predicted alpha.
            meta (list[dict]): Meta data about the current data batch.
                Currently only batch_size 1 is supported.

        Returns:
            np.ndarray: The reshaped predicted alpha.
        """
  
        ori_h, ori_w = meta[0]['merged_ori_shape'][:2]

        result = cv2.resize(result[0].cpu().numpy().transpose(1, 2, 0), (ori_w, ori_h), cv2.INTER_LANCZOS4) 

        assert result.shape[:2] == (ori_h, ori_w)

        return result

    def evaluate(self, pred_alpha, meta):
        """Evaluate predicted alpha matte.

        The evaluation metrics are determined by ``self.test_cfg.metrics``.

        Args:
            pred_alpha (np.ndarray): The predicted alpha matte of shape (H, W).
            meta (list[dict]): Meta data about the current data batch.
                Currently only batch_size 1 is supported. Required keys in the
                meta dict are ``ori_alpha`` and ``copy_trimap``.

        Returns:
            dict: The evaluation result.
        """
        if self.test_cfg.metrics is None:
            return None

        ori_alpha = meta[0]['ori_alpha'].squeeze()
        ori_trimap = meta[0]['copy_trimap'].squeeze()   # Note: copy_trimap is the origin np trimap 0-255

        eval_result = dict()
        for metric in self.test_cfg.metrics:
            eval_result[metric] = self.allowed_metrics[metric](
                ori_alpha, ori_trimap,
                np.round(pred_alpha * 255).astype(np.uint8))
        return eval_result
