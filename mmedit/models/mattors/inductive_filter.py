import torch
import mmcv
import cv2
import numpy as np
from pathlib import Path
import os
import torch.nn.functional as F
from ..builder import build_loss, build_component
from ..registry import MODELS
from ..common import set_requires_grad
from .base_mattor import BaseMattor

@MODELS.register_module()
class InductiveFilter(BaseMattor):
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
                 disc=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 loss_global=None,
                 loss_local=None,
                 loss_gabor=None,
                 loss_sync=None,
                 loss_gan=None,
                 loss_gp=None):
        super(InductiveFilter, self).__init__(backbone, None, train_cfg, test_cfg,
                                       pretrained)
        self.with_gan = disc is not None and loss_gan is not None
        self.with_gp_loss = loss_gp is not None
        self.loss_global = (
            build_loss(loss_global) if loss_global is not None else None)
        self.loss_local = (
            build_loss(loss_local) if loss_local is not None else None)
        self.loss_gabor = (
            build_loss(loss_gabor) if loss_gabor is not None else None)
        self.loss_sync = (
            build_loss(loss_sync) if loss_sync is not None else None)
        if self.with_gan:
            self.disc = build_component(disc)
            self.loss_gan = build_loss(loss_gan)
            self.disc_steps = 1 if self.train_cfg is None else self.train_cfg.get(
                'disc_steps', 1)
            self.disc_init_steps = (0 if self.train_cfg is None else
                                    self.train_cfg.get('disc_init_steps', 0))
            self.step_counter = 0
            self.disc.init_weights(pretrained=None)
        if self.with_gp_loss:
            self.loss_gp = build_loss(loss_gp)

    def forward_dummy(self, inputs):
        return self.backbone(inputs)

    def train_step(self, data_batch, optimizer):
        """Defines the computation and network update at every training call.

        Args:
            data_batch (torch.Tensor): Batch of data as input.
            optimizer (torch.optim.Optimizer): Optimizer of the model.

        Returns:
            dict: Output of ``train_step`` containing the logging variables \
                of the current data batch.
        """
        if not self.with_gan:
            outputs = self(**data_batch, test_mode=False)
            loss, log_vars = self.parse_losses(outputs.pop('losses')) 
            if "loss_sync" in log_vars:
                log_vars['loss_sync'] = log_vars['loss_sync'] * 100
            # optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            outputs.update({'log_vars': log_vars})
            return outputs

        log_vars = dict()
        outputs = self(**data_batch, test_mode=False)
        outputs['merged'] = data_batch['merged']
        outputs['mask'] = data_batch['mask']
        # get matting loss for g
        loss_matting, log_vars_matting = self.parse_losses(outputs.pop('losses'))
        log_vars.update(log_vars_matting)
        # optimize disc
        set_requires_grad(self.disc, True)
        optimizer['disc'].zero_grad()
        losses_disc = self.backward_disc(disc_input=outputs)
        loss_d, log_vars_d = self.parse_losses(losses_disc)
        log_vars.update(log_vars_d)
        loss_d.backward()
        optimizer['disc'].step()
        # optimize backbone, no updates to disc parameters.
        optimizer['backbone'].zero_grad()
        if (self.step_counter % self.disc_steps == 0
                 and self.step_counter >= self.disc_init_steps):
            set_requires_grad(self.disc, False)
            losses_backbone = self.backward_backbone(disc_input=outputs)
            loss_ganG, log_vars_g = self.parse_losses(losses_backbone)
            log_vars.update(log_vars_g)
            loss_g = loss_ganG + loss_matting
            loss_g.backward()
            optimizer['backbone'].step()
        else:
            loss_matting.backward()
            optimizer['backbone'].step()
        self.step_counter += 1

        log_vars.pop('loss', None)
        log_vars['loss'] = 0
        for key in log_vars.keys():
            log_vars['loss'] += log_vars[key]

        outputs.update({'log_vars': log_vars})
        del outputs['merged']
        del outputs['mask']
        del outputs['fake_alpha']
        del outputs['alpha']
        return outputs

    def forward_train(self, merged, mask, meta, alpha, local_mask, trimap, mask_2=None):
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
        pred_alpha = self.backbone(torch.cat((merged, mask), 1))
        if self.loss_sync is not None and mask_2 is not None:
            self.backbone.eval()
            pred_alpha_2 = self.backbone(torch.cat((merged, mask_2), 1))
            self.backbone.train()
            losses['loss_sync'] = self.loss_sync(pred_alpha, pred_alpha_2.detach()) 
        if self.loss_global is not None:
            losses['loss_global'] = self.loss_global(pred_alpha, alpha)
        if self.loss_local is not None:
            losses['loss_local'] = self.loss_local(pred_alpha, alpha, local_mask)
        if self.loss_gabor is not None:
            losses['loss_gabor'] = self.loss_gabor(pred_alpha, alpha)
        return {'losses': losses, 'num_samples': merged.size(0), 'fake_alpha': pred_alpha, 'alpha': alpha}

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
        mask = mask / mask.max()
        pred_alpha = self.backbone(torch.cat((merged, mask), 1))
        if isinstance(pred_alpha, tuple):
            pred_trimap = pred_alpha[1]
            pred_alpha = pred_alpha[0]
        pred_alpha = pred_alpha.cpu().numpy().squeeze()
        pred_alpha = np.clip(pred_alpha, 0, 1)
        meta[0]['ori_trimap'] = self.generate_trimap(meta[0]['ori_mask'])
        pred_alpha = self.restore_shape(pred_alpha, meta)
        meta[0]['ori_trimap'] = self.generate_pred_trimap(meta[0]['ori_mask'], pred_alpha, meta, save_path)
        meta[0]['ori_trimap'].fill(128)
        #pred_alpha[meta[0]['ori_trimap'] == 0] = 0.
        #pred_alpha[meta[0]['ori_trimap'] == 255] = 1.
        eval_result = self.evaluate(pred_alpha, meta)
    
        return {'pred_alpha': pred_alpha, 'eval_result': eval_result}

    def generate_pred_trimap(self, mask, pred_alpha, meta, save_path):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
        eroded = cv2.erode(mask, kernel, iterations=1)
        dilated = cv2.dilate(mask, kernel, iterations=1)
        trimap = np.zeros_like(pred_alpha)
        trimap.fill(128)
        trimap[eroded >= 1] = 255
        trimap[dilated <= 0] = 0

        x = cv2.Sobel(pred_alpha, cv2.CV_32F, 1, 0)
        y = cv2.Sobel(pred_alpha, cv2.CV_32F, 0, 1)
        dst = cv2.addWeighted(np.abs(x), 0.5, np.abs(y), 0.5, 0)
        dst[dst<0.2] = 0
        dst[dst>0.5] = 0
        dst = ((dst!=0)*255).astype(np.uint8)

        dilated = cv2.dilate(dst, kernel, iterations=1)
        trimap[dilated!=0] = 128
        
        if save_path is not None:
           image_stem = Path(meta[0]['merged_path']).stem
           save_path_p = os.path.join(save_path, f'{image_stem}.png')
           mmcv.imwrite(trimap.astype(np.uint8), save_path_p)
        return trimap

    def generate_trimap(self, mask):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                  (50, 50))
        eroded = cv2.erode(mask, kernel, iterations=1)
        dilated = cv2.dilate(mask, kernel, iterations=1)
        trimap = np.zeros_like(mask)
        trimap.fill(128)
        #trimap[eroded >= 1] = 255
        trimap[dilated <= 0] = 0
        return trimap

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

    def backward_disc(self, disc_input):
        """Backward function for the disc.

        Args:
            disc_input (dict): Dict of forward results.

        Returns:
            dict: Loss dict.
        """
        # GAN loss for the disc
        losses = dict()
        # conditional GAN
        fake_ab = torch.cat((disc_input['merged'], disc_input['mask'], disc_input['fake_alpha']), 1)
        fake_pred = self.disc(fake_ab.detach())
        losses['loss_gan_d_fake'] = self.loss_gan(
            fake_pred, target_is_real=False, is_disc=True)
        real_ab = torch.cat((disc_input['merged'], disc_input['mask'],disc_input['alpha']), 1)
        real_pred = self.disc(real_ab)
        losses['loss_gan_d_real'] = self.loss_gan(
            real_pred, target_is_real=True, is_disc=True)

        if self.with_gp_loss:
            loss_d_gp = self.loss_gp(
                self.disc, real_ab, disc_input['fake_alpha'], mask=None)
            losses['loss_gan_d_gp']  = loss_d_gp
        return losses

    def backward_backbone(self, disc_input):
        """Backward function for the backbone.

        Args:
            outputs (dict): Dict of forward results.

        Returns:
            dict: Loss dict.
        """
        losses = dict()
        # GAN loss for the backbone
        fake_ab = torch.cat((disc_input['merged'], disc_input['mask'], disc_input['fake_alpha']), 1)
        fake_pred = self.disc(fake_ab)
        losses['loss_gan_g'] = self.loss_gan(
            fake_pred, target_is_real=True, is_disc=False)
        return losses
