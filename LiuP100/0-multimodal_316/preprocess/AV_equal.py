'''
这个代码是来处理audio和video数据之间的不相等不平衡的问题
'''
import os

def av_number(audio_root,video_root):
    # 语音中188和316是多余的

    # audio_root='/usr/data/local/duanyuchi/python_data/multimodal/All_data/audio_s_c3_s1/'
    # video_root='/usr/data/local/duanyuchi/python_data/multimodal/All_data/video_c3_s1/'
    # av_number(audio_root,video_root)
    audio_list=[i for i in sorted(os.listdir(audio_root))]
    video_list=[j for j in sorted(os.listdir(video_root))]
    print(audio_list)
    print(video_list)
    for i in range(296):
        if audio_list[i]!=video_list[i]:
            print(audio_list[i])

def look_dir_num(root_audio,root_video):
    #看看所有数据中视频和音频的分段数量是不是一一对应的，都是这个可变的帧率带来的影响。
    # 结果表明，这个不平衡是切实存在的


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
            print('{}-{},A:{},V:{}'.format(id,dirs,num_wav,num_npy))
root_audio = '/usr/data/local/duanyuchi/python_data/multimodal/All_data/c3_s3/audio_s_c3_s3/'
root_video = '/usr/data/local/duanyuchi/python_data/multimodal/All_data/c3_s3/video_c3_s3/'
# look_dir_num(root_audio, root_video)
def look_data_balance():
    '''
    查看数据的信息
    man:111,woman:185
    dep:159,non:137
    '''
    root='/usr/data/local/duanyuchi/python_data/multimodal/All_data/c3_s1/video_c3_s1/'
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

#查看txt文件中是否平衡
'''
def txt_balance(txt):
    dep=0
    non=0
    with open(txt,'r') as rf:
        lines=rf.readlines()
    for line in lines:
        line=line.rstrip()
        label = line.split(";")[2]
        if label == '0':
            non += 1
        elif label == '1':
            dep += 1
    print("dep:{},non:{}".format(dep, non))
txt_dir="/usr/data/local/duanyuchi/python_data/multimodal/All_data/c3_s3/"
for txt in sorted(os.listdir(txt_dir)):
    if txt.endswith('txt'):
        print(txt,end=" ")
        txt_balance(os.path.join(txt_dir,txt))
'''

