import cv2
import os
import numpy as np
from tqdm import tqdm

def change_bg(img_dir, alpha_dir, save_dir):
    '''
    给原图换成白色背景，根据alpha合成图片
    '''

    img_lists = os.listdir(img_dir)

    bar = tqdm(img_lists)

    for img_name in bar:
        img_path = os.path.join(img_dir, img_name)
        alpha_path = os.path.join(alpha_dir, img_name)

        img = cv2.imread(img_path)
        alpha = cv2.imread(alpha_path, 0)[:, :, None]
        alpha = alpha / 255.0
        # alpha[alpha<0.05] = 0
        # alpha[alpha>0.95] = 1
        withe_bg = np.full_like(img, 255)
        white_bg = img * alpha + withe_bg * (1 - alpha)

        cv2.imwrite(os.path.join(save_dir, img_name), white_bg)

    print('change bg and merged finish')

def compose_adobe_val(fg_dir, alpha_dir, save_dir):
    fg_list = os.listdir(fg_dir)
    alpha_list = os.listdir(alpha_dir)

    for fg_name in tqdm(fg_list):
        fg_pre_name = os.path.splitext(fg_name)[0] # 文件名
        for alpha_name in tqdm(alpha_list):
            if(alpha_name.find(fg_pre_name) != -1):
                fg = cv2.imread(os.path.join(fg_dir, fg_name))
                alpha = cv2.imread(os.path.join(alpha_dir, alpha_name), 0)[:, :, None]
                alpha = alpha / 255.0
                withe_bg = np.full_like(fg, 0)
                white_bg = fg * alpha + withe_bg * (1 - alpha)

                cv2.imwrite(os.path.join(save_dir, alpha_name), white_bg)

        



if __name__ == "__main__":
    change_bg('data/GS_Video/man/results/port+purebg/fg', 'data/GS_Video/man/results/port+purebg/alpha', 'data/GS_Video/man/results/port+purebg/fg-merged')
    #change_bg('data/portrait/results/debug/port-withfusion-orifba-iter_44000.pth/fg', 'data/portrait/results/debug/port-withfusion-orifba-iter_44000.pth/alpha', 'data/portrait/results/debug/port-withfusion-orifba-iter_44000.pth/fg-merged')  
    # change_bg('data/portrait/images', 'data/portrait/results/debug-fb-no-sigmoid-iter_215440.pth/alpha', 'data/portrait/results/debug-fb-no-sigmoid-iter_215440.pth/img-merged')
    # change_bg('data/portrait/images', 'data/portrait/results/debug-portrait-only-iter-10000/alpha', 'data/portrait/results/debug-portrait-only-iter-10000/img-merged')
    # change_bg('data/GS_Video/results/ngs-iter_78000-SAD-7.081.pth/fg', 'data/GS_Video/results/ngs-iter_78000-SAD-7.081.pth/alpha', 'data/GS_Video/results/ngs-iter_78000-SAD-7.081.pth/fg-merged')
    #change_bg('data/Combined_Dataset/Test_set/Adobe-licensed images/fg', 'data/Combined_Dataset/Test_set/Adobe-licensed images/alpha', 'work_dirs/eval/fg-merged/gt-fg-merged')
    #compose_adobe_val('data/Combined_Dataset/Test_set/Adobe-licensed images/fg', 'work_dirs/eval/alpha/gca', 'work_dirs/eval/img-merged/gca')
    # change_bg('data/GS_Video/motion_blur/results/ngs-iter_78000-SAD-7.081.pth/fg', 'data/GS_Video/motion_blur/results/ngs-iter_78000-SAD-7.081.pth/alpha', 'data/GS_Video/motion_blur/results/ngs-iter_78000-SAD-7.081.pth/fg-merged')
    # change_bg('data/GS_Video/motion_blur/img', 'data/GS_Video/motion_blur/results/ngs-iter_78000-SAD-7.081.pth/alpha', 'data/GS_Video/motion_blur/results/ngs-iter_78000-SAD-7.081.pth/img-merged')
    # change_bg('data/GS_Video/man/ngs-iter_78000-SAD-7.081.pth/fg', 'data/GS_Video/man/ngs-iter_78000-SAD-7.081.pth/alpha-odf0.05', 'data/GS_Video/man/ngs-iter_78000-SAD-7.081.pth/fg-merged-ofd0.05')
    # change_bg('data/GS_Video/motion_blur/results/ngs-iter_78000-SAD-7.081.pth/fg-low', 'data/GS_Video/motion_blur/results/ngs-iter_78000-SAD-7.081.pth/alpha-ofd0.1', 'data/GS_Video/motion_blur/results/ngs-iter_78000-SAD-7.081.pth/fg-merged-ofd0.1')
