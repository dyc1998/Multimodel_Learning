import os
import shutil
import sys
import torchaudio
import matplotlib.pyplot as plt
def ChooseAuido18(root_source, root_out):
    '''
    这段代码之前只是为了提出第18的片段，但是现在需要对所有的片段都进行提取，[2,6,7,8,9,10,11,12,13,18,19]
    '''
    # 将18段的语音单独拿出来使用，重新建立一个新的文件夹来存放18的数据。
    dir_name = ['align_gansu2', 'align_guangyuan4', 'align_tianshui3']
    for i in range(3):
        source_dir = os.path.join(root_source, dir_name[i], 'audio')
        dirs = os.listdir(source_dir)
        for name_id in dirs:
            for wav in os.listdir(os.path.join(source_dir, name_id)):
                if wav[2:-4] == '-2':
                    out_dir=os.path.join(root_out, name_id,wav[:-6])
                    os.makedirs(out_dir)
                else:
                    out_dir = os.path.join(root_out, name_id, wav[:-6])
                    os.makedirs(out_dir)
                try:
                    shutil.copy(os.path.join(source_dir,name_id,wav),os.path.join(out_dir,wav))
                    print('name_id:',name_id)
                except Exception as e:
                    print(e)
    #  临时写的代码，
    # source_dir = os.path.join(root_source, 'video')
    # for dirs in os.listdir(source_dir):
    #     if dirs in ['1-071-07030710', '1-072-07030720', '1-073-07030730', '1-074-07030740', '1-075-07030750', '1-076-07030760',
    #                '1-077-07030770', '1-078-07030780', '1-079-07030790', '1-082-07030820', '1-083-07030830', '1-084-07030840',
    #                '1-085-07030850', '1-086-07030860', '1-087-07030870', '1-088-07030880', '1-089-07030890', '1-090-07030900',
    #                '1-091-07030910', '1-092-07030920', '1-093-07030930', '1-094-07030940', '1-095-07030950', '1-096-07030960',
    #                '1-097-07030970', '1-098-07030980', '1-099-07030990', '1-100-07031000', '1-101-07031010', '1-102-07031020',
    #                '1-103-07031030', '1-104-07031040', '1-105-07031050', '1-106-07031060', '1-107-07031070', '1-108-07031080',
    #                '1-109-07031090', '1-110-07031100', '1-111-07031110', '1-113-07031131']:
    #         for wav in os.listdir(os.path.join(source_dir, dirs)):
    #             if wav == '18-2.mp4':
    #                 out_dir=os.path.join(root_out, dirs)
    #                 os.mkdir(out_dir)
    #                 try:
    #                     shutil.copy(os.path.join(source_dir,dirs,wav),os.path.join(out_dir,wav))
    #                     print('name_id:',dirs)
    #                 except Exception as e:
    #                     print(e)
root_source = '/home/liuzhengyu/duanw/python_data/multimodal/'
root_out = '/home/liuzhengyu/duanw/python_data/multimodal/All_data/audio/'
# ChooseAuido18(root_source,root_out)




def SingleAduioExtract(root_source,root_single):
    for name in os.listdir(root_source):
        for wav_id in os.listdir(os.path.join(root_source,name)):
            for wav in os.listdir(os.path.join(root_source,name,wav_id)):
                wav_path = os.path.join(root_source, name, wav_id,wav)
                os.makedirs(os.path.join(root_single,name,wav_id))
                wav_single_path=os.path.join(root_single,name,wav_id,wav)
                waveform, sr = torchaudio.load(wav_path)
                if name[6:8] in ['05','06']:
                    waveform=waveform[1].unsqueeze(0)
                elif name[6:8] in ['07']:
                    waveform = waveform[0].unsqueeze(0)
                torchaudio.save(wav_single_path, waveform, sr)
root_source='/home/liuzhengyu/duanw/python_data/multimodal/All_data/audio/'
root_single='/home/liuzhengyu/duanw/python_data/multimodal/All_data/audio_single/'
# SingleAduioExtract(root_source,root_single)

def InfoCheck(root_audio,img_root=None):
    '''
    这个函数很有用，不仅帮我找到了一些数据空白的错误，而且还帮我找出来了sr不一致的问题。
    '''
    name_id=os.listdir(root_audio)
    print(name_id)
    for name in name_id:
        for wav_id in os.listdir(os.path.join(root_audio,name)):
            print(name)
            for wav in os.listdir(os.path.join(root_audio,name,wav_id)):
                waveform,sr=torchaudio.load(os.path.join(root_audio,name,wav_id,wav))

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
                # fbank_delta = torchaudio.functional.compute_deltas(fbank)
                # Compute fbank delta delta features
                # fbank_delta_delta = torchaudio.functional.compute_deltas(fbank_delta)
                fbank = fbank.transpose(0, 1)
                # fbank_delta=fbank_delta.transpose(0,1)
                # fbank_delta_delta=fbank_delta_delta.transpose(0,1)
                # plt.imshow(fbank)
                plt.title('{}-{}'.format(name,wav))
                # plt.imshow(fbank_delta)
                # plt.title('2')
                # plt.imshow(fbank_delta_delta)
                # plt.title('3')
                print(fbank.shape)
                # img = torch.stack((fbank, fbank_delta, fbank_delta_delta), dim=0)
                # img=img.permute(1,2,0)
                plt.imshow(fbank)




                plt.savefig(os.path.join(img_root, "{}-{}.png".format(name,wav)))
                plt.show()
root_audio='/home/liuzhengyu/duanw/python_data/multimodal/All_data/audio_single/'
img_root='/home/liuzhengyu/duanw/python_data/multimodal/All_data/AudioFbank/'
InfoCheck(root_audio,img_root)
sys.exit()


def ChooseVideo18(root_source, root_out):
    #将18段的语音单独拿出来使用，重新建立一个新的文件夹来存放18的数据。
    dir_name = ['align_gansu2', 'align_guangyuan4', 'align_tianshui3']
    for i in range(3):
        source_dir = os.path.join(root_source, dir_name[i], 'video')
        dirs = os.listdir(source_dir)
        for name_id in dirs:
            for wav in os.listdir(os.path.join(source_dir, name_id)):
                if wav[2:-4] == '-2':
                    out_dir = os.path.join(root_out, name_id, wav[:-6])
                    os.makedirs(out_dir)
                else:
                    out_dir = os.path.join(root_out, name_id, wav[:-6])
                    os.makedirs(out_dir)
                try:
                    shutil.copy(os.path.join(source_dir, name_id, wav), os.path.join(out_dir, wav))
                    print('name_id:', name_id)
                except Exception as e:
                    print(e)
root_source = '/home/liuzhengyu/duanw/python_data/multimodal/'
root_out = '/home/liuzhengyu/duanw/python_data/multimodal/All_data/video/'
# ChooseVideo18(root_source,root_out)