import argparse

import mmcv

def parse_args():
    parser = argparse.ArgumentParser(
        description='Cut Video Frames',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('video_root', help='video data root')
    parser.add_argument('mask_root', help='mask data root')
    return parser.parse_args()

def main():
    args = parse_args()
    if not osp.exists(args.data_root):
        raise FileNotFoundError(f'{args.data_root} does not exist!')

    print('generating Background Matting dataset annotation file...')
    generate_json(args.data_root, args.seg_root, args.bg_root, args.all_data)
    print('annotation file generated...')


if __name__ == '__main__':
    main()