import mmcv
import os
import cv2
import numpy as np

def cut_video(video_dir, save_dir):
    video_list = os.listdir(video_dir)

    for video_name in video_list:
        video = mmcv.VideoReader(os.path.join(video_dir, video_name))
        video.cvt2frames(save_dir, filename_tmpl=video_name[:-4] + '_{:06d}.png')
        

def concat_video(video_1, video_2, video_3, video_save):
    video1 = cv2.VideoCapture(video_1)
    video2 = cv2.VideoCapture(video_2)
    video3 = cv2.VideoCapture(video_3)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    video_writer = cv2.VideoWriter(os.path.join(video_save, 'EIG-vs-Last-vs-Our.avi'), fourcc,  30, (5760, 1080))
    

    i = 0

    while(1):
        a1, b1 = video1.read()
        a2, b2 = video2.read()
        a3, b3 = video3.read()
        if not a1  or not a2 or not a3:
            break
        # b1 = np.rot90(b1, -1)
        # b2 = np.rot90(b2, -1)
        # b3 = np.rot90(b3,-1)


        #b1 = np.rot90(b1, -1)
        c = np.concatenate((b1,b2,b3), 1)
        video_writer.write(c.astype(np.uint8))
        i+=1
        print(i)

def concat_2video(video_1, video_2, video_save):
    video1 = cv2.VideoCapture(video_1)
    video2 = cv2.VideoCapture(video_2)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    video_writer = cv2.VideoWriter(os.path.join(video_save, 'fg-merged-VS-ofd0.1.avi'), fourcc,  30, (3840, 1080))
    

    i = 0

    while(1):
        a1, b1 = video1.read()
        a2, b2 = video2.read()
        #a3, b3 = video3.read()
        if not a1  or not a2:
            break
        #b1 = np.rot90(b1, -1)
        #b2 = np.rot90(b2, -1)
        # b3 = np.rot90(b3,-1)


        #b1 = np.rot90(b1, -1)
        c = np.concatenate((b1,b2), 1)
        video_writer.write(c.astype(np.uint8))
        i+=1
        print(i)




if __name__ == "__main__":
   # cut_video('data/green-video', 'data/green-video/green-background')

    # video = mmcv.VideoReader('data/GS_Video/VID_20200727_140923.mp4')
    # video.cvt2frames('data/GS_Video/gsvideo_image', filename_tmpl='{:06d}.png')

    # mmcv.frames2video('data/GS_Video/results/ngs-iter_78000-SAD-7.081.pth/fg-merged', 'data/GS_Video/results/ngs-iter_78000-SAD-7.081.pth/fg-merged.avi', filename_tmpl='{:06d}.png')

    #concat_video('data/GS_Video/results/ngs-iter_78000-SAD-7.081.pth/img-merged.avi', 'data/GS_Video/results/ngs-iter_78000-SAD-7.081.pth/fg-merged.avi', 'data/GS_Video/results/ngs-iter_78000-SAD-7.081.pth')

    # concat_video('data/GS_Video/results/ngs-iter_78000-SAD-7.081.pth/SDK_raw_index.avi', 'data/GS_Video/results/ngs-iter_78000-SAD-7.081.pth/EIG.mp4', 'data/GS_Video/results/ngs-iter_78000-SAD-7.081.pth/fg-merged.avi', 'data/GS_Video/results/ngs-iter_78000-SAD-7.081.pth/submit')

    # video = mmcv.VideoReader('data/GS_Video/motion_blur/sorce.avi')
    # video.cvt2frames('data/GS_Video/motion_blur/sorce', filename_tmpl='{:06d}.png')

    # mmcv.frames2video('data/GS_Video/motion_blur/results/ngs-iter_78000-SAD-7.081.pth/fg-merged', 'data/GS_Video/motion_blur/results/ngs-iter_78000-SAD-7.081.pth/fg-merged/fg-merged.avi', filename_tmpl='{:06d}.png')
    # mmcv.frames2video('data/GS_Video/motion_blur/results/ngs-iter_78000-SAD-7.081.pth/img-merged', 'data/GS_Video/motion_blur/results/ngs-iter_78000-SAD-7.081.pth/fg-merged/img-merged.avi', filename_tmpl='{:06d}.png')
    # mmcv.frames2video('data/GS_Video/motion_blur/results/ngs-iter_78000-SAD-7.081.pth/fg-merged-low', 'data/GS_Video/motion_blur/results/ngs-iter_78000-SAD-7.081.pth/fg-merged-low.avi', filename_tmpl='{:06d}.png')

    # concat_video('data/GS_Video/motion_blur/results/compare/IMG_1967_EIG.mp4', 'data/GS_Video/motion_blur/results/compare/IMG_1967_indexnet_ft_result.avi', 'data/GS_Video/motion_blur/results/ngs-iter_78000-SAD-7.081.pth/fg-merged-low.avi', 'data/GS_Video/motion_blur/results/compare')

    mmcv.frames2video('data/GS_Video/man/results/port+purebg/fg-merged', 'data/GS_Video/man/results/port+purebg/port+purebg-epoch30.avi', filename_tmpl='{:06d}.png')

    # concat_2video('data/GS_Video/motion_blur/results/ngs-iter_78000-SAD-7.081.pth/fg-merged-low.avi', 'data/GS_Video/motion_blur/results/ngs-iter_78000-SAD-7.081.pth/ofd0.1.avi', 'data/GS_Video/motion_blur/results/ngs-iter_78000-SAD-7.081.pth')

    # cut_video('data/green-video/video/other_color', 'data/portrait+purebg/bg')