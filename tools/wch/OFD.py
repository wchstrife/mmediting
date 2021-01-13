import cv2
import numpy as np
import os
from tqdm import tqdm

def ofd(alpha_dir, save_dir):
    alpha_list = os.listdir(alpha_dir)
    alpha_list = sorted(alpha_list)

    alpha_pre = None
    alpha_now = None
    alpha_last = None
    
    for alpha_name in tqdm(alpha_list):
        alpha = cv2.imread(os.path.join(alpha_dir, alpha_name), 0) / 255
        if alpha_pre is None:
            alpha_pre = alpha
            cv2.imwrite(os.path.join(save_dir, alpha_name), alpha*255)
            continue
        if alpha_now is None:
            alpha_now = alpha
            cv2.imwrite(os.path.join(save_dir, alpha_name), alpha*255)
            continue

        alpha_last = alpha
            
        value = 0.1
        index = np.where( (np.abs(alpha_last - alpha_pre) <= value) & (np.abs(alpha_now - alpha_pre) > value) & (np.abs(alpha_last - alpha_now) > value) )
        alpha[index] = (alpha_pre[index] + alpha_last[index] / 2)

        alpha_pre = alpha_now
        alpha_now = alpha_last

        cv2.imwrite(os.path.join(save_dir, alpha_name), alpha*255)

def test():
    alpha_pre = np.array([[0,0],[0,0]])
    alpha_now = np.array([[0.5,0.5],[0.3,0.3]])
    alpha_last = np.array([[0,0],[1,1]])
    value = 0.1
    print(alpha_now)
    index = np.where( (np.abs(alpha_last - alpha_pre) <= value) & (np.abs(alpha_now - alpha_pre) > value) & (np.abs(alpha_last - alpha_now) > value) )
    alpha_now[index] = (alpha_pre[index] + alpha_last[index] / 2)
    print(alpha_now)

if __name__ == "__main__":
    #ofd('data/GS_Video/man/ngs-iter_78000-SAD-7.081.pth/alpha', 'data/GS_Video/man/ngs-iter_78000-SAD-7.081.pth/fg-merged-odf0.05')
    #ofd('data/GS_Video/motion_blur/results/ngs-iter_78000-SAD-7.081.pth/alpha-low', 'data/GS_Video/motion_blur/results/ngs-iter_78000-SAD-7.081.pth/alpha-ofd0.1')
    #test()

