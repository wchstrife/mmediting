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
        """Defines the computation performed at every test call.

        Args:
            image (Tensor): Image to predict alpha matte.
            two_chan_trimap (Tensor): Trimap of the input image(2 channels).
            image_n (Tensor):
            trimap_transformed(Tensor): 
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

        resnet_input = torch.cat((image_n, trimap_transformed, two_chan_trimap), 1)
        result = self.backbone(resnet_input, image)
        
        # TODO: 分离Decoder的输出
        pred_alpha = result.cpu().numpy().squeeze()
        pred_alpha = self.restore_shape(pred_alpha, meta)
        eval_result = self.evaluate(pred_alpha, meta)

        if save_image:
            self.save_image(pred_alpha, meta, save_path, iteration)

        return {'pred_alpha': pred_alpha, 'eval_result': eval_result}
        
