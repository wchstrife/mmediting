import argparse
import os
import mmcv
import torch


from mmedit.apis import init_model, matting_inference


def parse_args():
    parser = argparse.ArgumentParser(description='Matting demo')
    parser.add_argument('--config', help='test config file path', default='configs/mattors/fba/fba_comp1k_check.py')
    parser.add_argument('--checkpoint', help='checkpoint file', default='work_dirs/fba/model/iter_530000_36.645.pth')
    parser.add_argument('--img_path', help='path to input image file', default='data/alphamatting/input_lowres')
    parser.add_argument('--trimap_path', help='path to input trimap file', default='data/alphamatting/trimap_lowres/Trimap3')
    parser.add_argument('--save_path', help='path to save alpha matte result', default='data/alphamatting/result/Trimap3')
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

    for img in img_dir:
        pred_alpha = matting_inference(model, os.path.join(args.img_path, img),
                                   os.path.join(args.trimap_path, img)) * 255

        mmcv.imwrite(pred_alpha, os.path.join(args.save_path, img))


if __name__ == '__main__':
    main()
