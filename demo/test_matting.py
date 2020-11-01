import argparse

import mmcv
import torch

from mmedit.apis import init_model

from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint

from mmedit.datasets.pipelines import Compose
from mmedit.models import build_model


def parse_args():
    parser = argparse.ArgumentParser(description='Matting demo')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('img_path', help='path to input image file')
    parser.add_argument('trimap_path', help='path to input trimap file')
    parser.add_argument('save_path', help='path to save alpha matte result')
    parser.add_argument(
        '--imshow', action='store_true', help='whether show image with opencv')
    parser.add_argument('--device', type=int, default=0, help='CUDA device id')
    args = parser.parse_args()
    return args

def matting_inference(model, img, trimap):
    """Inference image(s) with the model.

    Args:
        model (nn.Module): The loaded model.
        img (str): Image file path.
        trimap (str): Trimap file path.

    Returns:
        np.ndarray: The predicted alpha matte.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # remove alpha from test_pipeline
    keys_to_remove = ['alpha', 'ori_alpha']
    for key in keys_to_remove:
        for pipeline in list(cfg.test_pipeline):
            if 'key' in pipeline and key == pipeline['key']:
                cfg.test_pipeline.remove(pipeline)
            if 'keys' in pipeline and key in pipeline['keys']:
                pipeline['keys'].remove(key)
                if len(pipeline['keys']) == 0:
                    cfg.test_pipeline.remove(pipeline)
            if 'meta_keys' in pipeline and key in pipeline['meta_keys']:
                pipeline['meta_keys'].remove(key)
    # build the data pipeline
    test_pipeline = Compose(cfg.test_pipeline)
    # prepare data
    data = dict(merged_path=img, trimap_path=trimap)
    data = test_pipeline(data)

    # # Test Code
    # merged = data['merged'] 
    # ori_merged = data['ori_merged']
    # trimap = data['trimap']
    # trimap_transformed = data['trimap_transformed']

    # ori_merged.cpu().numpy().tofile('dat/' + 'ori_merged_new' + '.dat')
    # merged.cpu().numpy().tofile('dat/' + 'merged_rgbtrue' + '.dat')
    # trimap.cpu().numpy().tofile('dat/' + 'trimap' + '.dat')
    # trimap_transformed.numpy().tofile('dat/' + 'trimap_transformed' + '.dat')

    data = scatter(collate([data], samples_per_gpu=1), [device])[0]

    # merged = data['merged'] 
    # merged.cpu().numpy().tofile('dat/merged_tensor_new_norm_list.dat')
    print("data prepare success!!!")
    # forward the model
    with torch.no_grad():
        result = model(test_mode=True, **data)

    return result['pred_alpha']

def rename_pth(checkpoint):
    from collections import OrderedDict

    fy = torch.load(checkpoint)
    if 'state_dict' in fy:
        state_dict = fy['state_dict']
    else:
        state_dict = fy

    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        name = k.replace('encoder', 'encoder.encoder')
        name = 'backbone.' + name
        new_state_dict[name] = v

    for i in new_state_dict.keys():
        print(i, list(new_state_dict[i].size()))
    
    torch.save(new_state_dict, 'work_dirs/fba/FBA_rename_pat.pth')

def main():
    args = parse_args()

    rename_pth(args.checkpoint)
    print('rename success')

    # model = init_model(
    #     args.config, args.checkpoint, device=torch.device('cuda', args.device))

    # for i in model.state_dict():
    #     print(i)

    # pred_alpha = matting_inference(model, args.img_path,
    #                                args.trimap_path) * 255

    # # print(pred_alpha)
    # mmcv.imwrite(pred_alpha, args.save_path)
    # if args.imshow:
    #     mmcv.imshow(pred_alpha, 'predicted alpha matte')


if __name__ == '__main__':
    main()

