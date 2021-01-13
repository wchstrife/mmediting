import cv2
import os
import numpy as np

def check_trimap(trimap_dir):
    trimap_list = os.listdir(trimap_dir)

    for trimap_name in trimap_list:
        trimap = cv2.imread(os.path.join(trimap_dir, trimap_name), 0) 
        ll = np.unique(trimap).tolist()
        if len(ll) !=3:
            print(ll)
            print(trimap_name)
        # trimap[trimap==0] = 0
        # trimap[trimap==255] = 0
        # trimap[trimap==128] = 0

        # trimap[trimap!=0] = 1
        
        #print(np.sum(trimap == 1))
    #for each in tqdm(files):
    # im = cv2.imread(each, 0)
    # ll = np.unique(im).tolist()
    # if len(ll) !=3:
    #     print(ll)
    #     print(each)

def format_triamp(trimap_dir, save_dir):
    trimap_list = os.listdir(trimap_dir)

    for trimap_name in trimap_list:
        trimap = cv2.imread(os.path.join(trimap_dir, trimap_name), 0) 
        trimap_fix = np.full_like(trimap, 128)
        trimap_fix[trimap<128] = 0
        trimap_fix[trimap>128] = 255
        cv2.imwrite(os.path.join(save_dir, trimap_name), trimap_fix)

        ll = np.unique(trimap_fix).tolist()
        if len(ll) !=3:
            print(ll)
            print(trimap_name)

if __name__ == "__main__":
    check_trimap('data/portrait/trimaps-fix')
    #format_triamp('data/portrait/trimaps', 'data/portrait/trimaps-fix')