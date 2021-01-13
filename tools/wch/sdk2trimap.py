import cv2
import numpy as np
import os

def mask2trimap(alpha_dir, save_dir):
    # SDK生成的mask变成trimap
    alpha_list = os.listdir(alpha_dir)

    for alpha_name in alpha_list:
        alpha = cv2.imread(os.path.join(alpha_dir, alpha_name), 0)
        alpha_blur = cv2.blur(alpha, (8,8))
        temp = alpha - alpha_blur
        alpha = 255 - alpha
        alpha[alpha > 128] = 255
        alpha[alpha < 128] = 0
        alpha[temp > 0] = 128

        kernel = np.ones((5,5), np.uint8) 
        alpha = cv2.morphologyEx(alpha, cv2.MORPH_OPEN, kernel)
        alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, kernel)

        cv2.imwrite(os.path.join(save_dir, alpha_name), alpha)

def fixtrimap(trimap_dir, save_dir):
    trimap_list = os.listdir(trimap_dir)

    for alpha_name in trimap_list:
        alpha = cv2.imread(os.path.join(trimap_dir, alpha_name), 0)
        alpha_blur = cv2.blur(alpha, (8,8))
        temp = alpha - alpha_blur
        alpha[alpha > 128] = 255
        alpha[alpha < 128] = 0
        alpha[temp > 0] = 128

        kernel = np.ones((5,5), np.uint8) 
        alpha = cv2.morphologyEx(alpha, cv2.MORPH_OPEN, kernel)
        alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, kernel)

        cv2.imwrite(os.path.join(save_dir, alpha_name[:-4]+'.png'), alpha)


if __name__ == "__main__":
    mask2trimap('data/GS_Video/motion_blur/mask', 'data/GS_Video/motion_blur/trimap-mask-low')
    #fixtrimap('data/portrait/trimaps', 'data/portrait/trimaps-fix')