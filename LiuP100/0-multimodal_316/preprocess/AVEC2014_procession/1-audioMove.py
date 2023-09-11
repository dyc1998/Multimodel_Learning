import os
import shutil

'''
主要将avec2014的数据处理成相同的数据格式
'''
def audioMove(source_root,end_root):
    for dtt in os.listdir(os.path.join(source_root)):
        if(dtt=='label'):
            continue
        os.mkdir(os.path.join(end_root,dtt))
        for fn in os.listdir(os.path.join(source_root,dtt)):

            for wav in os.listdir(os.path.join(source_root,dtt,fn)):
                if wav.endswith(".wav"):
                    shutil.copy(os.path.join(source_root, dtt, fn, wav), os.path.join(end_root, dtt))


source_root="/usr/data/local/duanyuchi/python_data/AVEC_2014/source_data/";
end_root="/usr/data/local/duanyuchi/python_data/AVEC_2014/1-Audio/"
audioMove(source_root,end_root)