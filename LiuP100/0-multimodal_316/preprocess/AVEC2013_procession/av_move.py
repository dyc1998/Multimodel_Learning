import os
import shutil

'''
主要将avec2013的数据处理成相同的数据格式
'''
def audioMove(v_dep,source_audio,end_audio):
    for mp4 in os.listdir(os.path.join(v_dep)):
        shutil.move(os.path.join(source_audio,mp4[:-3]+'wav'),os.path.join(end_audio))

            # for wav in os.listdir(os.path.join(source_root,dtt,fn)):
            #     if wav.endswith(".wav"):
            #         shutil.copy(os.path.join(source_root, dtt, fn, wav), os.path.join(end_root, dtt))

v_dep='/usr/data/local/duanyuchi/python_data/AVEC_2103/1-Video/Development/'
source_root="/usr/data/local/duanyuchi/python_data/AVEC_2103/1-Audio/Train/"
end_root="/usr/data/local/duanyuchi/python_data/AVEC_2103/1-Audio/Development/"
audioMove(v_dep,source_root,end_root)