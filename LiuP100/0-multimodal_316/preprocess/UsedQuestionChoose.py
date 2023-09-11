'''
这段代码主要是来从天水，甘肃2和广元4中的三家医院中，进行数据选择
目前的想法是只使用第18段语音和视频，主要是觉得18段的长度和说话方式比较固定，操作起来好弄，
1-对应选择语音的函数：ChooseAuido18(root_source, root_out):
2-对应选择视频的函数：ChooseVideo18(root_source, root_out):
3-查看一下这些语音数据的质量和并且统计一下比如sr等具体的情况.发现了一些sr的问题和左右声道的问题:InfoCheck(root_audio,img_root):
4-直接把语音数据中单个声道的有用数据保存成csv文件吧,不然之后的预处理就麻烦了
'''
import os
import sys
import torch
import cv2
import torchaudio
import shutil
import matplotlib.pyplot as plt
import os
from pydub.silence import split_on_silence
from pydub import AudioSegment, silence
import pydub
from moviepy.editor import VideoFileClip
from torchaudio.functional import compute_deltas

from torchvision import transforms


def renames(root):
    # 给一些之前数据错误的文件进行重命名。
    for dirs in os.listdir(root):
        if dirs in ['1-071-07030710', '1-072-07030720', '1-073-07030730', '1-074-07030740', '1-075-07030750', '1-076-07030760',
                   '1-077-07030770', '1-078-07030780', '1-079-07030790', '1-082-07030820', '1-083-07030830', '1-084-07030840',
                   '1-085-07030850', '1-086-07030860', '1-087-07030870', '1-088-07030880', '1-089-07030890', '1-090-07030900',
                   '1-091-07030910', '1-092-07030920', '1-093-07030930', '1-094-07030940', '1-095-07030950', '1-096-07030960',
                   '1-097-07030970', '1-098-07030980', '1-099-07030990', '1-100-07031000', '1-101-07031010', '1-102-07031020',
                   '1-103-07031030', '1-104-07031040', '1-105-07031050', '1-106-07031060', '1-107-07031070', '1-108-07031080',
                   '1-109-07031090', '1-110-07031100', '1-111-07031110', '1-113-07031131']:
            os.rename(os.path.join(root, dirs), os.path.join(root, '3' + dirs[1:]))
root_1='/home/liuzhengyu/duanw/python_data/multimodal/ThreeOnly18/video/'
# renames(root_1)







def InfoCheck(root_audio,img_root=None):
    '''
    这个函数很有用，不仅帮我找到了一些数据空白的错误，而且还帮我找出来了sr不一致的问题。
    '''
    name_id=os.listdir(root_audio)
    print(name_id[0])
    for name in [name_id[0]]:
        print(name)
        for wav in os.listdir(os.path.join(root_audio,name)):
            waveform,sr=torchaudio.load(os.path.join(root_audio,name,wav))

            waveform = waveform[:,0:5 * sr]
            # if len(waveform[0])/sr>45:
            #     print('{}:{:.3f}'.format(name,len(waveform[0])/sr))
            # waveform = waveform.unsqueeze(0)
            # if sr==48000:
            #     print('{}:{}'.format(name,sr))

            fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
                                                      frame_length=40,
                                                      window_type='hanning', num_mel_bins=64, dither=0.0,
                                                      frame_shift=20)

            # Compute fbank delta features
            fbank_delta = torchaudio.functional.compute_deltas(fbank)
            # Compute fbank delta delta features
            fbank_delta_delta = torchaudio.functional.compute_deltas(fbank_delta)
            fbank = fbank.transpose(0, 1)
            fbank_delta=fbank_delta.transpose(0,1)
            fbank_delta_delta=fbank_delta_delta.transpose(0,1)
            # plt.imshow(fbank)
            # plt.title('1')
            # plt.imshow(fbank_delta)
            # plt.title('2')
            # plt.imshow(fbank_delta_delta)
            # plt.title('3')
            print(fbank.shape)
            img = torch.stack((fbank, fbank_delta, fbank_delta_delta), dim=0)
            img=img.permute(1,2,0)

            plt.imshow(img)




            plt.savefig(os.path.join(img_root, f"0-{name}_3dFbank.png"))
            plt.show()
root_audio='/home/liuzhengyu/duanw/python_data/multimodal/ThreeOnly18/audio_single/'
img_root='/home/liuzhengyu/duanw/python_data/multimodal/ThreeOnly18/audioFbank/'
# InfoCheck(root_audio,img_root)

# 查看视频的信息，分辨率
def InfoCheckVideo(root_video):
    # Load video file
    for name_id in os.listdir(root_video):
        for mp4 in os.listdir(os.path.join(root_video,name_id)):
            cap = cv2.VideoCapture(os.path.join(root_video,name_id,mp4))
            # Get video resolution
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"{name_id}: {width}x{height}")

root_video='/home/liuzhengyu/duanw/python_data/multimodal/ThreeOnly18/video/'
# InfoCheckVideo(root_video)







def lenth_compare(root_audio,root_video):
    sum=0
    list1=[]
    for i in os.listdir(root_audio):
        sum+=1
        for wav in os.listdir(os.path.join(root_audio,i)):
            try:
                sour_audio=os.path.join(root_audio,i,wav)
                sound = pydub.AudioSegment.from_wav(sour_audio)
                wav_length = sound.duration_seconds
                mp4 = wav[:-4] + '.mp4'
                sour_video=os.path.join(root_video,i,mp4)
                clip = VideoFileClip(sour_video)
                mp4_length = clip.duration
                # if int(abs(mp4_length - wav_length)) < 1:
                    # print('{}/{}'.format(i, mp4))
                    # print('OK,{}'.format(sum))
                if i in ['1-159-05031591', '1-123-05031230', '1-148-05031480', '1-047-05030471', '1-074-05030740', '1-120-05011201', '1-169-05031690', '1-144-05011440', '1-002-05010021', '1-161-05031610', '1-044-05030441', '1-075-05010751', '1-119-05031191', '1-175-05031750', '1-177-05031771', '1-040-05010400', '2-070-05010701', '1-176-05011760', '1-076-05010760', '1-157-05031571', '1-195-05011950', '1-151-05031510', '1-126-05011261', '1-124-05031240', '1-130-05031301', '1-066-05010661', '1-183-05031830', '1-064-05030640', '1-057-05010571', '1-145-05031451', '1-182-05031821', '1-055-05010550', '1-174-05031741', '1-036-05030361', '1-131-05031311', '1-073-05010730', '1-170-05031701', '1-146-05031460', '1-050-05010501', '1-060-05030601', '1-135-05011351', '1-051-05010510', '1-025-05010251', '1-173-05031730', '1-049-05010491', '1-139-05011390', '1-154-05011541', '1-136-05031361', '1-052-05010520', '1-147-05031470', '1-129-05031291', '1-188-05031881', '1-121-05031210', '1-053-05010531', '1-187-05031870', '1-165-05031651', '1-138-05011380', '1-127-05031271', '1-140-05031400', '1-072-05030721', '1-160-05031600']:

                    if int(abs(mp4_length*1.2 - wav_length))>1:
                        print('{}/{} has question,A{},V{}'.format(i, wav, wav_length, mp4_length))
                        list1.append(i)
                clip.close()
            except Exception as e:
                print(e)
    print(list1)

# root_audio='/home/liuzhengyu/duanw/python_data/multimodal/ThreeOnly18/auido/'
# root_video='/home/liuzhengyu/duanw/python_data/multimodal/ThreeOnly18/video/'
# lenth_compare(root_audio,root_video)

def look_dir_num(root_audio,root_video):
    #看看视频和音频的分段数量是不是一一对应的，都是这个可变的帧率带来的影响。
    wav_list=[]
    npy_list=[]
    for id in os.listdir(root_audio):
        num_wav = 0
        num_npy = 0
        for dirs in os.listdir(os.path.join(root_audio,id)):
            wav=os.listdir(os.path.join(root_audio,id,dirs))
            num_wav=len(wav)
            wav_list.append(int(num_wav))
        for dirs in os.listdir(os.path.join(root_video,id)):
            mp4=os.listdir(os.path.join(root_video,id,dirs))
            num_npy=len(mp4)
            npy_list.append(int(num_npy))
        if num_npy !=num_wav:
            print('{},A:{},V:{}'.format(id,num_wav,num_npy))

root_audio='/home/liuzhengyu/duanw/python_data/multimodal/ThreeOnly18/auido_s_c3s_s1s/'
root_video='/home/liuzhengyu/duanw/python_data/multimodal/ThreeOnly18/video_c3s_s1s/'
# look_dir_num(root_audio,root_video)

def look_data_balance():
    root='/home/liuzhengyu/duanw/python_data/multimodal/ThreeOnly18/video/'
    man=0
    woman=0
    dep=0
    non=0
    for name_id in os.listdir(root):
        if name_id[8:10]=='03':
            non+=1
        else:
            dep+=1
        if name_id[-1]=='0':
            woman+=1
        else:
            man+=1
    print('man:{},woman:{}'.format(man,woman))
    print('dep:{},non:{}'.format(dep,non))
# look_data_balance()