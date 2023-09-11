# -*- coding: utf-8 -*-
'''
2023年3月29日，对自己数据的处理，
处理思路是将我们自己的视频数据变成只有人脸的图片，最后按照序列打包。
测试了cv库，dlib库进行裁剪人脸，但是发现裁剪的都不好，而且也不全面，所以还是用openface那个工具去裁剪。
使用openface进行裁剪，必须要调用cmd，但是linux系统操作起来很麻烦，所以就只能在本地跑了，在我的小新上跑的。
我在本地使用了这段代码的代码如下
'''
import math
import os
import subprocess
import sys
import torchaudio
import numpy as np
import cv2 as cv
# from tqdm import tqdm
# import torch
# from facenet_pytorch import MTCNN
# import cv2
# from torchvision import transforms


# TODO 将视频裁剪成只有人脸的图片,在本地的小新的jupyter上进行运行
def face_crop(file_path, output_path):
    '''
    file_path = r'I:\多模态数据\ThreeOnly18\video'
    output_path = r'I:\多模态数据\ThreeOnly18\video_crop_img'
    我在本地使用了这段代码的代码如下
    '''
    exe_path = r'D:\OpenFace_2.2.0_win_x64\FeatureExtraction.exe'
    i = 1
    for nameid in os.listdir(file_path):
        for dirs in os.listdir(os.path.join(file_path, nameid)):
            out_file = os.path.join(output_path, nameid, dirs[:4])

            video_path = os.path.join(file_path, nameid, dirs)
            params0 = exe_path[
                      1:] + ' -f ' + video_path + ' -out_dir ' + out_file + ' -aligned png -simsize 224  -simalign -2Dfp -pose -gaze -au'
            print(i, params0)
            i = i + 1
            print('开始裁剪：' + video_path)
        subprocess.run(params0)
# 对处理好的npy文件发现有些是18-2，所以需要修改一下命名，让这个都统一
def renamesss():
    root_video='/usr/data/local/duanyuchi/python_data/multimodal/All_data/video2img/'
    for name_id in os.listdir(root_video):
        for dirs_id in os.listdir(os.path.join(root_video,name_id)):
            for dirs in os.listdir(os.path.join(root_video,name_id,dirs_id)):
                # print(dirs)
                if dirs[2:4]=='-2':
                    print(name_id)
                    os.rename(os.path.join(root_video,name_id,dirs_id,dirs),os.path.join(root_video,name_id,dirs_id,dirs[:3]+'1'+dirs[4:]))
# renamesss()

def seq_npy_creat(root,root_audio, root_out, seq=6,target_length=3, shift_step=3):
    '''
    生成seq图片的npy文件
    target_length=3,目标长度为3s，
    shift_step=1，每一秒选一下。
    这个代码把我写麻了，但是我觉得我也学到了很多，这个代码让我对如何处理多模态数据有了更加深刻的见解。
    '''
    target_seq=target_length*seq
    for dirs in sorted(os.listdir(root)):
        if dirs not in['1-181-07011810']:
            continue
        print(dirs)
        os.mkdir(os.path.join(root_out,dirs))
        seq_num = []
        for wav_id in os.listdir(os.path.join(root, dirs)):
            # if wav_id not in['19']:
            #     continue
            os.mkdir(os.path.join(root_out, dirs,wav_id))
            count=0
            img_path=sorted(os.listdir(os.path.join(root, dirs, wav_id, wav_id + '-1_aligned')))
            wavs=sorted(os.listdir(os.path.join(root_audio,dirs,wav_id)))
            for wav in wavs:
                sound,sr=torchaudio.load(os.path.join(root_audio,dirs,wav_id,wav))

                wav_length = sound.shape[1]/sr
                img_num=len(img_path)
                fps=round(img_num/wav_length)  #对应的图片数量除以音频的长度，得到视频本身的帧率
                # gap=round((fps/6.0)/2)

            num_segments = math.ceil((img_num - target_length*fps) / (fps*shift_step))+1
            for i in range(num_segments):
                start=i*shift_step*fps
                end=start+target_length*fps
                if end > len(img_path):
                    end=len(img_path)
                for im in range(start,end):
                    if (im+1)%round(fps/6)==0:
                        img=cv.imread(os.path.join(root,dirs,wav_id,wav_id + '-1_aligned',img_path[im]))
                        seq_num.append(img)
                if len(seq_num) >target_seq:
                    np.save(os.path.join(root_out, dirs, wav_id, wav_id + '_{:03d}.npy'.format(i)),np.array(seq_num[:18]))
                    count += 1
                    seq_num = []
                elif len(seq_num) == target_seq:
                    np.save(os.path.join(root_out, dirs, wav_id, wav_id + '_{:03d}.npy'.format(i)),
                            np.array(seq_num))
                    count += 1
                    seq_num = []
                elif 0 < len(seq_num) < target_seq:  # 如果frames_to_select这个列表还有东西，
                    # 那表明处理的图片不够18个，需要补0，使得最终的numpy_video一共有15个图片
                    for j in range(18 - len(seq_num)):
                        seq_num.append(np.zeros((224, 224, 3), dtype=np.uint8))
                    np.save(os.path.join(root_out, dirs, wav_id, wav_id + '_{:03d}.npy'.format(i)),
                            np.array(seq_num))
                    count+=1
                    seq_num = []
            # for i in range(start,end):
            #
            #     batch_seq=img_path[i+1:i+target_length*fps:round(fps/6)]
            #     for imgs in batch_seq:
            #         img = cv.imread(os.path.join(root,dirs,wav_id,wav_id + '_aligned',imgs))
            #         seq_num.append(img)
            #     if len(seq_num) >target_seq:
            #         np.save(os.path.join(root_out, dirs, wav_id, wav_id + '_{:03d}.npy'.format(i)),np.array(seq_num[:18]))
            #         count += 1
            #         seq_num = []
            #     elif len(seq_num) == target_seq:
            #         np.save(os.path.join(root_out, dirs, wav_id, wav_id + '_{:03d}.npy'.format(i)),
            #                 np.array(seq_num))
            #         count += 1
            #         seq_num = []
            #     elif 0 < len(seq_num) < target_seq:  # 如果frames_to_select这个列表还有东西，
            #         # 那表明处理的图片不够18个，需要补0，使得最终的numpy_video一共有15个图片
            #         for j in range(18 - len(seq_num)):
            #             seq_num.append(np.zeros((224, 224, 3), dtype=np.uint8))
            #         np.save(os.path.join(root_out, dirs, wav_id, wav_id + '_{:03d}.npy'.format(i)),
            #                 np.array(seq_num))
            #         count+=1
            #         seq_num = []





            # if dirs in ['1-159-05031591', '1-123-05031230', '1-148-05031480', '1-047-05030471', '1-074-05030740',
            #             '1-120-05011201', '1-169-05031690', '1-144-05011440', '1-002-05010021', '1-161-05031610',
            #             '1-044-05030441', '1-075-05010751', '1-119-05031191', '1-175-05031750', '1-177-05031771',
            #             '1-040-05010400', '2-070-05010701', '1-176-05011760', '1-076-05010760', '1-157-05031571',
            #             '1-195-05011950', '1-151-05031510', '1-126-05011261', '1-124-05031240', '1-130-05031301',
            #             '1-066-05010661', '1-183-05031830', '1-064-05030640', '1-057-05010571', '1-145-05031451',
            #             '1-182-05031821', '1-055-05010550', '1-174-05031741', '1-036-05030361', '1-131-05031311',
            #             '1-073-05010730', '1-170-05031701', '1-146-05031460', '1-050-05010501', '1-060-05030601',
            #             '1-135-05011351', '1-051-05010510', '1-025-05010251', '1-173-05031730', '1-049-05010491',
            #             '1-139-05011390', '1-154-05011541', '1-136-05031361', '1-052-05010520', '1-147-05031470',
            #             '1-129-05031291', '1-188-05031881', '1-121-05031210', '1-053-05010531', '1-187-05031870',
            #             '1-138-05011380', '1-127-05031271', '1-140-05031400', '1-072-05030721', '1-160-05031600']:
            #
            #     for i in range(0,len(img_path),shift_step*24):
            #         batch_seq=img_path[i:i+target_length*24:4]
            #         for imgs in batch_seq:
            #             img = cv.imread(imgs)
            #             seq_num.append(img)
            #         if len(seq_num) == target_seq:
            #             np.save(os.path.join(root_out, dirs, wav_id, wav_id + '_{:03d}.npy'.format(count)),
            #                     np.array(seq_num))
            #             count += 1
            #             seq_num = []
            #         elif 0 < len(seq_num) < target_seq:  # 如果frames_to_select这个列表还有东西，
            #             # 那表明处理的图片不够18个，需要补0，使得最终的numpy_video一共有15个图片
            #             for j in range(18 - len(seq_num)):
            #                 seq_num.append(np.zeros((224, 224, 3), dtype=np.uint8))
            #             np.save(os.path.join(root_out, dirs, wav_id, wav_id + '_{:03d}.npy'.format(count)),
            #                     np.array(seq_num))
            #             seq_num = []
            # else:
            #     for i in range(0,len(img_path),shift_step*30):
            #         batch_seq=img_path[i:i+target_length*30:5]
            #         for imgs in batch_seq:
            #             img = cv.imread(imgs)
            #             seq_num.append(img)
            #         if len(seq_num) == target_seq:
            #             np.save(os.path.join(root_out, dirs, wav_id, wav_id + '_{:03d}.npy'.format(count)),
            #                     np.array(seq_num))
            #             count += 1
            #             seq_num = []
            #         elif 0 < len(seq_num) < target_seq:  # 如果frames_to_select这个列表还有东西，
            #             # 那表明处理的图片不够18个，需要补0，使得最终的numpy_video一共有15个图片
            #             for j in range(18 - len(seq_num)):
            #                 seq_num.append(np.zeros((224, 224, 3), dtype=np.uint8))
            #             np.save(os.path.join(root_out, dirs, wav_id, wav_id + '_{:03d}.npy'.format(count)),
            #                     np.array(seq_num))
            #             seq_num = []

root = '/usr/data/local/duanyuchi/python_data/multimodal/All_data/video2img/'
root_out = '/usr/data/local/duanyuchi/python_data/multimodal/All_data/c3_s3/try/'
root_audio = '/usr/data/local/duanyuchi/python_data/multimodal/All_data/audio_single/'
seq_npy_creat(root, root_audio,root_out)



# img_root='/home/liuzhengyu/duanw/python_data/multimodal/ThreeOnly18/video_crop_img/1-008-07030081/18-1/18-1_aligned/frame_det_00_000002.bmp'
# img_root2='/home/liuzhengyu/duanw/python_data/TC-data/dataset/negative/002-05010021_aligned/frame_det_00_000001.png'
# img=cv.imread(img_root)
# print(img.shape)
# print(img[100:,100:,:])