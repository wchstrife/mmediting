import torch
import torch.nn as nn
import torch.nn.functional as F

import cv2
import numpy as np
from ..registry import LOSSES
from .utils import masked_loss

_reduction_modes = ['none', 'mean', 'sum']

@masked_loss
def mse_loss(pred, target):
    """MSE loss.

    Args:
        pred (Tensor): Prediction Tensor with shape (n, c, h, w).
        target ([type]): Target Tensor with shape (n, c, h, w).

    Returns:
        Tensor: Calculated MSE loss.
    """
    return F.mse_loss(pred, target, reduction='none')

@LOSSES.register_module()
class GaborLoss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        sample_wise (bool): Whether calculate the loss sample-wise. This
            argument only takes effect when `reduction` is 'mean' and `weight`
            (argument of `forward()`) is not None. It will first reduce loss
            with 'mean' per-sample, and then it means over all the samples.
            Default: False.
    """

    def __init__(self, filter_size, theta, Lambda, sigma, gamma, loss_weight=1.0, reduction='mean', sample_wise=False):
        super(GaborLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')
        assert theta, 'theta should not be empty.'
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.sample_wise = sample_wise
        self.kernels = self.getGaborKernels(filter_size, theta, Lambda, sigma, gamma)

    def getGaborKernels(self, filter_size, theta, Lambda, sigma, gamma):
        kernels = []
        for th in theta:
            kernel_para = cv2.getGaborKernel(filter_size, sigma, th, Lambda, gamma, ktype=cv2.CV_32F)
            kernel_para = nn.Parameter(torch.from_numpy(kernel_para).view(1, 1, 5, 5))
            kernel_para.requires_grad=False
            filter = nn.Conv2d(1, 1, 5, 1, 1, bias=False)
            filter.weight = kernel_para
            kernels.append(filter.cuda())
        return kernels

    def forward(self, pred, target, weight=None, **kwargs):
        """Forward Function.

        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        loss = 0

        for kernel in self.kernels:
            p = kernel(pred)
            t = kernel(target)
            loss += mse_loss(
                        p,
                        t,
                        weight,
                        reduction=self.reduction,
                        sample_wise=self.sample_wise)
        return self.loss_weight * loss
