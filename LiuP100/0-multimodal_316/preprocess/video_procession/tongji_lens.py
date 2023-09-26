#统计每一个任务的长度，最后去做截断到平均长度，不够的补0，最后按照一个人为单位，将所有任务的数据进行拼接，得到以人为基准的数据。
import os
import numpy as np

my_dict = {}
def stastic_people_len(source_root):
    for people in os.listdir(source_root):
        for ids in os.listdir(os.path.join(source_root,people)):
            sec=np.load(os.path.join(source_root,people,ids,ids+'.npy')).shape[0]
            if ids in my_dict:
                my_dict[ids].append(sec)
            else:
                my_dict[ids]=[sec]
    for key in sorted(my_dict.keys()):
        print(key,':',np.mean(my_dict[key]))
def cat_all(source_root,out_root,my_dict):
    people_all={}
    for people in os.listdir(source_root):
        for ids in os.listdir(os.path.join(source_root,people)):
            npy=np.load(os.path.join(source_root,people,ids,ids+'.npy'))

            cut2avg=npy[:round(np.mean(my_dict[ids])),:]
            cutandpad2avg = np.pad(cut2avg, ((0,  round(np.mean(my_dict[ids]))- cut2avg.shape[0]), (0, 0)), mode='constant',constant_values=0)
            # print("ids:",ids)
            # print("npy:",npy.shape)
            # print(round(np.mean(my_dict[ids])))
            # print("cut",cut2avg.shape)
            # print("cut_pad",cutandpad2avg)
            if people in people_all:
                people_all[people].append(cutandpad2avg)
            else:
                people_all[people]=[cutandpad2avg]
    for peop in people_all.keys():
        os.mkdir(os.path.join(out_root, peop))
        np.save(os.path.join(out_root, peop,peop + '.npy'), np.concatenate(people_all[peop],axis=0))

        #测试一下
        # C=np.concatenate(people_all[peop],axis=0)
        # print(type(C)==type(np.array(C)))






if __name__ == '__main__':
    source_root = '/usr/data/local/duanyuchi/python_data/multimodal/All_data/video_136/'
    out_root = '/usr/data/local/duanyuchi/python_data/multimodal/All_data/video_people/'
    # stastic_people_len(source_root)
    # cat_all(source_root,out_root,my_dict)

    #平均每个人的视频是（156，136）的维度，意思是156s的平均长度
    data=np.load('/usr/data/local/duanyuchi/python_data/multimodal/All_data/video_people/1-014-07010141/1-014-07010141.npy',allow_pickle=True)
    # data1=np.load('/usr/data/local/duanyuchi/python_data/multimodal/All_data/video_136/1-008-07030081/09/09.npy')
    print(data.shape)

