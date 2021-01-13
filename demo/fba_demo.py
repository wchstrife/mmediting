import argparse
import os

import mmcv
import torch

from mmedit.apis import init_model, fba_inference


def parse_args():
    parser = argparse.ArgumentParser(description='Matting demo')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('img_path', help='path to input image ')
    parser.add_argument('trimap_path', help='path to input trimap ')
    parser.add_argument('save_path', help='path to save alpha matte result')
    parser.add_argument(
        '--imshow', action='store_true', help='whether show image with opencv')
    parser.add_argument('--device', type=int, default=1, help='CUDA device id')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    model = init_model(
        args.config, args.checkpoint, device=torch.device('cuda', args.device))

    img_dir = os.listdir(args.img_path)

    for img in img_dir:

        pred_alpha, perd_fg, pred_bg = fba_inference(model, os.path.join(args.img_path, img) , os.path.join(args.trimap_path, img))

        mmcv.imwrite(pred_alpha * 255, os.path.join('data/portrait/results/debug-nograd-noexcel/alpha', img))
        mmcv.imwrite(perd_fg * 255, os.path.join('data/portrait/results/debug-nograd-noexcel/fg', img))
        # mmcv.imwrite(pred_bg * 255, os.path.join('data/portrait/results/debug-portrait-only/bg', img))



if __name__ == '__main__':
    main()
