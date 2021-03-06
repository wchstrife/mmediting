import argparse
import os
import mmcv
import torch
from tqdm import tqdm


from mmedit.apis import init_model, matting_inference


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


def main():
    args = parse_args()

    model = init_model(
        args.config, args.checkpoint, device=torch.device('cuda', args.device))

    img_dir = os.listdir(args.img_path)

    for img in tqdm(img_dir):
        pred_alpha = matting_inference(model, os.path.join(args.img_path, img),
                                   os.path.join(args.trimap_path, img)) * 255

        mmcv.imwrite(pred_alpha, os.path.join("data/GS_Video/results/ngs-iter_78000-SAD-7.081.pth/alpha", img))


if __name__ == '__main__':
    main()
