import mmcv
import numpy as np

from ..registry import PIPELINES


@PIPELINES.register_module()
class Normalize(object):
    """Normalize images with the given mean and std value.

    Required keys are the keys in attribute "keys", added or modified keys are
    the keys in attribute "keys" and these keys with postfix '_norm_cfg'.
    It also supports normalizing a list of images.

    Args:
        keys (Sequence[str]): The images to be normalized.
        mean (np.ndarray): Mean values of different channels.
        std (np.ndarray): Std values of different channels.
        to_rgb (bool): Whether to convert channels from BGR to RGB.
    """

    def __init__(self, keys, mean, std, to_rgb=False):
        self.keys = keys
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        for key in self.keys:
            if isinstance(results[key], list):
                results[key] = [
                    mmcv.imnormalize(v, self.mean, self.std, self.to_rgb)
                    for v in results[key]
                ]
            else:
                results[key] = mmcv.imnormalize(results[key], self.mean,
                                                self.std, self.to_rgb)

        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(keys={self.keys}, mean={self.mean}, std={self.std}, '
                     f'to_rgb={self.to_rgb})')

        return repr_str


@PIPELINES.register_module()
class RescaleToZeroOne(object):
    """Transform the images into a range between 0 and 1.

    Required keys are the keys in attribute "keys", added or modified keys are
    the keys in attribute "keys".
    It also supports rescaling a list of images.

    Args:
        keys (Sequence[str]): The images to be transformed.
    """

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        for key in self.keys:
            if isinstance(results[key], list):
                results[key] = [
                    v.astype(np.float32) / 255. for v in results[key]
                ]
            else:
                results[key] = results[key].astype(np.float32) / 255.
                #results[key] = results[key] / 255.0
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(keys={self.keys})'

@PIPELINES.register_module()
class GroupNoraliseImage(object):
    """Normalize images with the given mean and std value.

    For FBA

    Args:
        keys (Sequence[str]): The images to be normalized.
        mean (np.ndarray): Mean values of different channels.
        std (np.ndarray): Std values of different channels.
    """

    def __init__(self, keys, mean, std, format,to_rgb=False):
        self.keys = keys
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.format = format
        self.to_rgb = to_rgb

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """

        for key in self.keys:
            img = results[key].clone()
            # img = img[None, :, :, :].float()
            # img.cpu().numpy().tofile('/home2/wangchenhao/mmediting/dat/before_norm.dat')
            if (format == 'hwc'):
                for i in range(3):
                    img[..., i] = (img[..., i] - self.mean[i]) / self.std[i]
            else:
                for i in range(3):
                    img[..., i, :, :] = (img[..., i, :, :] - self.mean[i]) / self.std[i]
            # img.cpu().numpy().tofile('/home2/wangchenhao/mmediting/dat/after_norm.dat')
            # img = img.squeeze()
            results[key] = img

        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(keys={self.keys}, mean={self.mean}, std={self.std}')

        return repr_str        
