# -*- coding: utf-8 -*-
# 2023-3-28xiugai
#这段代码是处理语音数据，将长语音数据严格的切割在2s,3s,4s,没有进行叠加抽取。少的补0，这个soundfile库需要注意一下。
import math
import sys

import librosa
import os
import soundfile as sf
import numpy as np
from tqdm import tqdm
import torchaudio

# 将18-2这样的片段变成18-1
def renamesss():
    root_video='/usr/data/local/duanyuchi/python_data/multimodal/All_data/audio_single/'
    for name_id in os.listdir(root_video):
        for dirs_id in os.listdir(os.path.join(root_video,name_id)):
            for dirs in os.listdir(os.path.join(root_video,name_id,dirs_id)):
                print(dirs[:3])
                if dirs[-6:-4]=='-2':
                    print(name_id)
                    os.rename(os.path.join(root_video,name_id,dirs_id,dirs),os.path.join(root_video,name_id,dirs_id,dirs[:3]+'1.wav'))
# renamesss()


#这个仅仅是第18段语音的代码，全部语音的代码需要重新写
# def auido_crop(source_root,target_root,target_time=3,shift_step=1):
#     for actor in os.listdir(source_root):
#         print(actor)
#         os.mkdir(os.path.join(target_root, actor))
#         for audiofile in os.listdir(os.path.join(source_root, actor)):
#             os.mkdir(os.path.join(target_root,actor,audiofile[:-4]))
#             if not audiofile.endswith('.wav') or 'croppad' in audiofile:
#                 continue
#             y,sr=torchaudio.load(os.path.join(source_root,actor,audiofile))
#
#             wav_length=y.shape[1]
#             target_length = int(sr * target_time)
#             shift_length=int(sr* shift_step)
#
#
#
#             # Get number of segments
#             num_segments = math.ceil((len(y[0])-target_length) / shift_length)+1
#             # Export each segment
#             for i in range(num_segments):
#                 start = i * shift_length
#                 end = start+target_length
#                 print(start*29)
#                 if end>=wav_length:
#                     end=wav_length
#                     # segment = y[:, start:end]
#                     # torchaudio.save(
#                     #     os.path.join(target_root, actor, audiofile[:-4], audiofile[:-4] + '_{:03d}.wav'.format(i * 29)),
#                     #     segment, sr)
#                     # break
#                 segment = y[:, start:end]
#                 torchaudio.save(os.path.join(target_root, actor, audiofile[:-4],audiofile[:-4] + '_{:03d}.wav'.format(i)), segment, sr)

# 这个是所有的语音片段处理的代码
def auido_crop_all(source_root,target_root,target_time=3,shift_step=3):
    for actor in os.listdir(source_root):
        print(actor)
        os.mkdir(os.path.join(target_root, actor))
        for audiofile_dir in os.listdir(os.path.join(source_root, actor)):
            os.mkdir(os.path.join(target_root,actor,audiofile_dir))
            for audiofile in os.listdir(os.path.join(source_root,actor,audiofile_dir)):

                if not audiofile.endswith('.wav') or 'croppad' in audiofile:
                    continue
                y,sr=torchaudio.load(os.path.join(source_root,actor,audiofile_dir,audiofile))

                wav_length=y.shape[1]
                target_length = int(sr * target_time)
                shift_length=int(sr* shift_step)



                # Get number of segments
                num_segments = math.ceil((len(y[0])-target_length) / shift_length)+1
                # Export each segment
                for i in range(num_segments):
                    start = i * shift_length
                    end = start+target_length
                    print(start*29)
                    if end>=wav_length:
                        end=wav_length
                        # segment = y[:, start:end]
                        # torchaudio.save(
                        #     os.path.join(target_root, actor, audiofile[:-4], audiofile[:-4] + '_{:03d}.wav'.format(i * 29)),
                        #     segment, sr)
                        # break
                    segment = y[:, start:end]
                    torchaudio.save(os.path.join(target_root, actor, audiofile_dir,audiofile[:-4] + '_{:03d}.wav'.format(i)), segment, sr)

source_root = '/usr/data/local/duanyuchi/python_data/multimodal/All_data/audio_single/'
target_root = '/usr/data/local/duanyuchi/python_data/multimodal/All_data/c3_s3/audio_s_c3_s3/'
auido_crop_all(source_root,target_root,target_time=3,shift_step=3)
