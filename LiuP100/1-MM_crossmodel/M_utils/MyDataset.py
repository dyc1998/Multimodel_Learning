# -*- coding: utf-8 -*-
"""
多模态的dataset
"""
import sys

import matplotlib.pyplot as plt
import torch
import torchaudio
import torch.utils.data as data
from PIL import Image
import functools
import numpy as np
import librosa
import transforms as transform
from torchvision import transforms


# 用来加载视频数据，将视频分帧之后，固定长度的选择，最后拼接起来，论文中说的是取15张图片
def video_loader(video_dir_path):
    video = np.load(video_dir_path)
    # print('111',video.shape) #(18, 224, 224, 3)
    video_data = []
    for i in range(np.shape(video)[0]):
        video_data.append(Image.fromarray(video[i, :, :, :]))  # 频数据从 numpy.array 转换为 Python 的 PIL.Image 类型

    # print('333-',len(video_data))
    # plt.imshow(video_data[0])
    # plt.show()
    return video_data


# 把上面的函数封装一下
def get_default_video_loader():
    '''
    get_default_video_loader 函数是对 video_loader 函数的封装。
    它返回一个使用 functools.partial 封装后的 video_loader 函数，
    意味着当调用该函数时，可以省略 video_dir_path 参数。
    '''
    # functools.partial 函数可以为函数提供定义好的输入变量，就像是函数的预定义变量一样，
    return functools.partial(video_loader)


# 加载语音数据，计算语音的长度和sr
def load_audio(audiofile, sr):
    audios = librosa.core.load(audiofile, sr=sr)
    y = audios[0]
    return y, sr


# 得出语音数据的mfcc
def get_mfccs(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=10)
    print('222', mfcc.shape)  # [10,130]
    return mfcc


def wav2fbank(filename):
    waveform, sr = torchaudio.load(filename)
    # 有些语音切完之后都不足一个帧长，所以会报错，至少补0到一个帧长的长度
    # print('waveform',waveform.shape)
    if waveform.shape[1]<1920:
        pp=1920-waveform.shape[1]
        m = torch.nn.ZeroPad2d((0, pp))
        waveform=m(waveform)
        # print('waveform后的',waveform.shape)


    waveform = waveform - waveform.mean()
    fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
                                              frame_length=40,
                                              window_type='hanning', num_mel_bins=128, dither=0.0,
                                              frame_shift=20)
    # print('fbank',fbank.shape)
    # print('333',fbank.shape)   #[149, 64]
    target_length = 149
    n_frames = fbank.shape[0]
    p = target_length - n_frames
    # cut or pad
    if p > 0:
        m = torch.nn.ZeroPad2d((0, 0, 0, p))
        fbank = m(fbank)
    elif p < 0:
        fbank = fbank[0:target_length, :]
    return fbank


def mask_Fbank(fbank, freqm=5, timem=5, skip_norm=False):
    freqm = torchaudio.transforms.FrequencyMasking(freqm)
    timem = torchaudio.transforms.TimeMasking(timem)
    fbank = torch.transpose(fbank, 0, 1)
    # this is just to satisfy new torchaudio version, which only accept [1, freq, time]
    fbank = fbank.unsqueeze(0)
    if freqm != 0:
        fbank = freqm(fbank)
    if timem != 0:
        fbank = timem(fbank)

    # squeeze it back, it is just a trick to satisfy new torchaudio version
    fbank = fbank.squeeze(0)
    fbank = torch.transpose(fbank, 0, 1)

    # 这个norm通过get_norm_state进行了计算
    # if skip_norm:
    #     fbank = (fbank - norm_mean) / (norm_std * 2)
    # else:
    #     pass
    return fbank


# 加载数据的路径和标签，返回一个保留路径和标签的列表，列表的元素是一个字典
def make_dataset(annotation_path):
    with open(annotation_path, 'r') as f:
        annots = f.readlines()

    dataset = []
    for line in annots:
        filename, audiofilename, label = line.split(';')
        sample = {'video_path': filename,
                  'audio_path': audiofilename,
                  'label': int(label)}
        dataset.append(sample)
    return dataset


class Multi_dataset(data.Dataset):  # 这里写的直接是一个多模态的dataloader
    def __init__(self,
                 annotation_path,  # 数据路径和标签的txt文件
                 video_transform=None,
                 audio_transform=None,
                 get_loader=get_default_video_loader,
                 data_type='audiovisual'):
        self.data = make_dataset(annotation_path)
        self.video_transform = video_transform
        self.audio_transform = audio_transform
        self.loader = get_loader()
        self.data_type = data_type

    def __getitem__(self, index):
        target = self.data[index]['label']

        if self.data_type == 'video' or self.data_type == 'audiovisual':
            path = self.data[index]['video_path']
            clip = self.loader(path)  # 长度是15,即15张3*224*224的RGB格式的数据Image.Image

            if self.video_transform is not None:
                self.video_transform.randomize_parameters()
                clip = [self.video_transform(img) for img in clip]
            '''
            这里的clip是由15个[3，224，224]的图片组成的列表，但是如何将这15个离散的图片
            变成一个整体，就使用了stack函数，将其变成了[15,3,224,224]
            然后经过permute变成了[3,15,224,224],所以clip的shape：【3，15，224，224】
            '''
            clip = torch.stack(clip, 0).permute(1, 0, 2, 3)  #

            if self.data_type == 'video':
                return clip, target

        if self.data_type == 'audio' or self.data_type == 'audiovisual':
            path = self.data[index]['audio_path']
            fbank = wav2fbank(path)
            if self.audio_transform is not None:
                fbank = mask_Fbank(fbank)
                # print(fbank.shape)
            fbank = fbank.transpose(1, 0)

            if self.data_type == 'audio':
                return fbank, target
        if self.data_type == 'audiovisual':
            return fbank, clip, target

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':

    annotation_path = '/usr/data/local/duanyuchi/python_data/multimodal/All_data/c3_s3/val_0.txt'
    video_transform = transform.Compose([
        transform.RandomHorizontalFlip(),
        transform.RandomRotate(),
        transform.ToTensor(255)])

    training_data = Multi_dataset(
        annotation_path,
        video_transform=video_transform,
        data_type='audiovisual',
        audio_transform=None)

    train_loader = torch.utils.data.DataLoader(
        training_data,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=False
    )

    for i, (audio_inputs, visual_inputs, targets) in enumerate(train_loader):
        print(i)
        print('audio', audio_inputs.shape)
        print('video', visual_inputs.shape)
        # img=visual_inputs[0].permute(1,2,3,0)
        # plt.imshow(img[0])
        # plt.show()
