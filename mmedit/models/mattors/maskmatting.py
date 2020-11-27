import torch
import mmcv
import cv2
import numpy as np
from pathlib import Path
import os
import torch.nn.functional as F
from ..builder import build_loss, build_component, build_backbone
from ..registry import MODELS
from ..common import set_requires_grad
from .base_mattor import BaseMattor
from .utils import get_unknown_tensor

@MODELS.register_module()
class MaskMatting(BaseMattor):
    """IndexNet matting model.

    This implementation follows:
    Indices Matter: Learning to Index for Deep Image Matting

    Args:
        backbone (dict): Config of backbone.
        train_cfg (dict): Config of training. In 'train_cfg', 'train_backbone'
            should be specified.
        test_cfg (dict): Config of testing.
        pretrained (str): path of pretrained model.
        loss_alpha (dict): Config of the alpha prediction loss. Default: None.
        loss_comp (dict): Config of the composition loss. Default: None.
    """

    def __init__(self,
                 backbone,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 loss_gabor=None,
                 loss_alpha=None,
                 loss_comp=None,
                 constraint_loss=None):
        super(MaskMatting, self).__init__(backbone, None, train_cfg, test_cfg,
                                       pretrained)
        self.loss_alpha = (
            build_loss(loss_alpha) if loss_alpha is not None else None)
        self.loss_comp = (
            build_loss(loss_comp) if loss_comp is not None else None)
        self.loss_gabor = (
            build_loss(loss_gabor) if loss_gabor is not None else None)
        self.constraint_loss=constraint_loss

    def forward_dummy(self, inputs):
        trimap_p, alpha_r = self.backbone(inputs)
        trimap_p = F.softmax(trimap_p, dim=1)
        _, unsure, fs = torch.split(trimap_p, 1, dim=1)
        alpha_p = fs+unsure*alpha_r
        return alpha_p

    def train_step(self, data_batch, optimizer):
        """Defines the computation and network update at every training call.

        Args:
            data_batch (torch.Tensor): Batch of data as input.
            optimizer (torch.optim.Optimizer): Optimizer of the model.

        Returns:
            dict: Output of ``train_step`` containing the logging variables \
                of the current data batch.
        """
        outputs = self(**data_batch, test_mode=False)
        loss, log_vars = self.parse_losses(outputs.pop('losses'))
        log_vars['loss_trimap'] = log_vars['loss_trimap'] * 100

        # optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        outputs.update({'log_vars': log_vars})
        return outputs


    def forward_train(self, merged, mask, meta, alpha, trimap, fg=None, bg=None, ori_merged=None):
        """Forward function for training IndexNet model.

        Args:
            merged (Tensor): Input images tensor with shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            trimap (Tensor): Tensor of trimap with shape (N, 1, H, W).
            meta (list[dict]): Meta data about the current data batch.
            alpha (Tensor): Tensor of alpha with shape (N, 1, H, W).
            ori_merged (Tensor): Tensor of origin merged images (not
                normalized) with shape (N, C, H, W).
            fg (Tensor): Tensor of foreground with shape (N, C, H, W).
            bg (Tensor): Tensor of background with shape (N, C, H, W).

        Returns:
            dict: Contains the loss items and batch infomation.
        """
        losses = dict()
        trimap_p, alpha_r = self.backbone(torch.cat((merged, mask), 1))
        trimap = trimap // 125
        losses['loss_trimap'] =  0.01 * F.cross_entropy(trimap_p, trimap.squeeze(1).long(), reduction='none').mean()
        trimap_p = F.softmax(trimap_p, dim=1)
        _, unsure, fs = torch.split(trimap_p, 1, dim=1)
        alpha_p = fs+unsure*alpha_r

        if self.constraint_loss == "groundtruth":
            if self.loss_alpha is not None:
                losses['loss_alpha'] = self.loss_alpha(alpha_r, alpha, ( alpha.lt(1) * alpha.gt(0) ).float())
            if self.loss_comp is not None:
                losses['loss_comp'] = self.loss_comp(alpha_r, fg, bg, ori_merged, ( alpha.lt(1) * alpha.gt(0) ).float())

        elif self.constraint_loss == "pred":
            trimap_index = torch.argmax(trimap_p, dim=1, keepdim=True)
            if self.loss_alpha is not None:
                losses['loss_alpha'] = self.loss_alpha(alpha_r, alpha, ( trimap_index.eq(1) ).float())
            if self.loss_comp is not None:
                losses['loss_comp'] = self.loss_comp(alpha_r, fg, bg, ori_merged, ( trimap_index.eq(1) ).float())

        else:
            if self.loss_alpha is not None:
                losses['loss_alpha'] = self.loss_alpha(alpha_p, alpha)
            if self.loss_comp is not None:
                losses['loss_comp'] = self.loss_comp(alpha_p, fg, bg, ori_merged)

        '''
        if self.loss_alpha is not None:
            losses['loss_alpha'] = self.loss_alpha(alpha_p, alpha)
        if self.loss_comp is not None:
            losses['loss_comp'] = self.loss_comp(alpha_p, fg, bg, ori_merged)
        if self.loss_gabor is not None:
            losses['loss_gabor'] = self.loss_gabor(alpha_p, alpha)
        '''        
        return {'losses': losses, 'num_samples': merged.size(0)}

    def forward_test(self,
                     merged,
                     mask,
                     meta,
                     save_image=False,
                     save_path=None,
                     iteration=None):
        """Defines the computation performed at every test call.

        Args:
            merged (Tensor): Image to predict alpha matte.
            trimap (Tensor): Trimap of the input image.
            meta (list[dict]): Meta data about the current data batch.
                Currently only batch_size 1 is supported. It may contain
                information needed to calculate metrics (``ori_alpha`` and
                ``ori_trimap``) or save predicted alpha matte
                (``merged_path``).
            save_image (bool, optional): Whether save predicted alpha matte.
                Defaults to False.
            save_path (str, optional): The directory to save predicted alpha
                matte. Defaults to None.
            iteration (int, optional): If given as None, the saved alpha matte
                will have the same file name with ``merged_path`` in meta dict.
                If given as an int, the saved alpha matte would named with
                postfix ``_{iteration}.png``. Defaults to None.

        Returns:
            dict: Contains the predicted alpha and evaluation result.
        """
        mask = mask/mask.max()
        trimap_p, alpha_r = self.backbone(torch.cat((merged, mask), 1))
        trimap_p = F.softmax(trimap_p, dim=1)
        _, unsure, fs = torch.split(trimap_p, 1, dim=1)
        alpha_p = fs + unsure * alpha_r
  
        pred_alpha = alpha_p.cpu().numpy().squeeze()
        pred_alpha = np.clip(pred_alpha, 0, 1)
        meta[0]['ori_trimap'] = np.zeros_like(meta[0]['ori_mask'])
        meta[0]['ori_trimap'].fill(128)
        
        pred_alpha = self.restore_shape(pred_alpha, meta)
        eval_result = self.evaluate(pred_alpha, meta)

        if save_image:
            self.save_image(pred_alpha, meta, save_path, iteration)

        return {'pred_alpha': pred_alpha, 'eval_result': eval_result}

    def forward(self,
                merged,
                mask,
                meta,
                alpha=None,
                test_mode=False,
                **kwargs):
        """Defines the computation performed at every call.

        Args:
            merged (Tensor): Image to predict alpha matte.
            trimap (Tensor): Trimap of the input image.
            meta (list[dict]): Meta data about the current data batch.
                Defaults to None.
            alpha (Tensor, optional): Ground-truth alpha matte.
                Defaults to None.
            test_mode (bool, optional): Whether in test mode. If ``True``, it
                will call ``forward_test`` of the model. Otherwise, it will
                call ``forward_train`` of the model. Defaults to False.

        Returns:
            dict: Return the output of ``self.forward_test`` if ``test_mode`` \
                are set to ``True``. Otherwise return the output of \
                ``self.forward_train``.
        """
        if not test_mode:
            return self.forward_train(merged, mask, meta, alpha, **kwargs)
        else:
            return self.forward_test(merged, mask, meta, **kwargs)
