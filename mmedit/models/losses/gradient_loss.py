import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
from torch import nn
import torch.nn.functional as fnn
from torch.autograd import Variable
from torch.optim import SGD
from torchvision.datasets import LSUN
from torchvision import transforms
from torch.utils.data import Dataset
from torchvision.utils import make_grid
import numpy as np

from ..registry import LOSSES
from .pixelwise_loss import l1_loss
from .utils import masked_loss

_reduction_modes = ['none', 'mean', 'sum']

@masked_loss
def l1_loss(pred, target):                                                                             
    """L1 loss.                                                                                       
                                                                                                      
    Args:                                                                                             
        pred (Tensor): Prediction Tensor with shape (n, c, h, w).                                      
        target ([type]): Target Tensor with shape (n, c, h, w).                                      
                                                                                                     
    Returns:                                                                                         
        Tensor: Calculated L1 loss.  
    """                                                               
    return F.l1_loss(pred, target, reduction='none') 


@LOSSES.register_module()
class LaplacianLoss(nn.Module):
    def __init__(self, max_levels=5, k_size=5, sigma=2.0, loss_weight=1.0, channel=1, cuda=True, reduction='mean'):
        super(LaplacianLoss, self).__init__()
        self.max_levels = max_levels
        self.k_size = k_size
        self.sigma = sigma
        self._gauss_kernel = self.build_gauss_kernel(
                size=self.k_size, sigma=self.sigma, 
                n_channels=channel, cuda=cuda
            )
        self.loss_weight = loss_weight
        self.reduction=reduction
        
    def forward(self, input, target, weight=None):
        pyr_input  = self.laplacian_pyramid( input, self._gauss_kernel, self.max_levels)
        pyr_target = self.laplacian_pyramid(target, self._gauss_kernel, self.max_levels)
        return self.loss_weight * (sum(l1_loss(a, b, weight, reduction=self.reduction, sample_wise=True) for a, b in zip(pyr_input, pyr_target)))

    def build_gauss_kernel(self, size=5, sigma=1.0, n_channels=1, cuda=False):
        if size % 2 != 1:
            raise ValueError("kernel size must be uneven")
        grid = np.float32(np.mgrid[0:size,0:size].T)
        gaussian = lambda x: np.exp((x - size//2)**2/(-2*sigma**2))**2
        kernel = np.sum(gaussian(grid), axis=2)
        kernel /= np.sum(kernel)
        # repeat same kernel across depth dimension
        kernel = np.tile(kernel, (n_channels, 1, 1))
        # conv weight should be (out_channels, groups/in_channels, h, w), 
        # and since we have depth-separable convolution we want the groups dimension to be 1
        kernel = torch.FloatTensor(kernel[:, None, :, :])
        if cuda:
            kernel = kernel.cuda()
        return Variable(kernel, requires_grad=False)


    def conv_gauss(self, img, kernel):
        """ convolve img with a gaussian kernel that has been built with build_gauss_kernel """
        n_channels, _, kw, kh = kernel.shape
        img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
        return F.conv2d(img, kernel, groups=n_channels)


    def laplacian_pyramid(self, img, kernel, max_levels=5):
        current = img
        pyr = []

        for level in range(max_levels):
            filtered = self.conv_gauss(current, kernel)
            diff = current - filtered
            pyr.append(diff)
            current = F.avg_pool2d(filtered, 2)

        pyr.append(current)
        return pyr


@LOSSES.register_module()
class GradientLoss(nn.Module):
    """Gradient loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(GradientLoss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
        if self.reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {self.reduction}. '
                             f'Supported ones are: {_reduction_modes}')

    def forward(self, pred, target, weight=None):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        kx = torch.Tensor([[1, 0, -1], [2, 0, -2],
                           [1, 0, -1]]).view(1, 1, 3, 3).to(target)
        ky = torch.Tensor([[1, 2, 1], [0, 0, 0],
                           [-1, -2, -1]]).view(1, 1, 3, 3).to(target)

        pred_grad_x = F.conv2d(pred, kx, padding=1)
        pred_grad_y = F.conv2d(pred, ky, padding=1)
        target_grad_x = F.conv2d(target, kx, padding=1)
        target_grad_y = F.conv2d(target, ky, padding=1)

        loss = (
            l1_loss(
                pred_grad_x, target_grad_x, weight, reduction=self.reduction) +
            l1_loss(
                pred_grad_y, target_grad_y, weight, reduction=self.reduction))
        return loss * self.loss_weight

@LOSSES.register_module()
class GradientExclusionLoss(nn.Module):
    """Gradient Exclusion Loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(GradientExclusionLoss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
        if self.reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {self.reduction}. '
                             f'Supported ones are: {_reduction_modes}')

    def forward(self, fg, bg, weight):
        """
        Args:
            fg (Tensor): of shape (N, C, H, W). FG tensor.
            bg (Tensor): of shape (N, C, H, W). BG tensor.
        """
        kx = torch.Tensor([[1, 0, -1], [2, 0, -2],
                           [1, 0, -1]]).view(1, 1, 3, 3).to(bg)
        ky = torch.Tensor([[1, 2, 1], [0, 0, 0],
                           [-1, -2, -1]]).view(1, 1, 3, 3).to(bg)

        fg_grad_x = F.conv2d(fg, kx, padding=1)
        fg_grad_y = F.conv2d(fg, ky, padding=1)
        bg_grad_x = F.conv2d(bg, kx, padding=1)
        bg_grad_y = F.conv2d(bg, ky, padding=1)

        fg_grad = torch.abs(fg_grad_x) + torch.abs(fg_grad_y)
        bg_grad = torch.abs(bg_grad_x) + torch.abs(bg_grad_y)

        grad = fg_grad.mul(bg_grad)
        zero = torch.zeros_like(grad).to(grad)

        loss = l1_loss(grad, zero, weight, reduction=self.reduction)

        return loss * self.loss_weight