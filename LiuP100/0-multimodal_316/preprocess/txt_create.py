import os
from sklearn.model_selection import train_test_split, KFold


def seq_npy_txtCreat(audio_path,video_path):
    '''
    生成npy文件的txt文件，用来训练
    '''
    # healthy = []
    # depression = []
    # # sample_rate = 60
    # # path = f'/home/liuzhengyu/duanw/python_data/TC-data/seq_npy_gap6/4s/{x[stimulate_index][1]}'
    #
    # sub_p = os.listdir(audio_path)
    # for s in sub_p:
    #     if s.split('_')[-1][-5] == '1':
    #         depression.append(s)
    #     else:
    #         healthy.append(s)
    # dep_n = len(depression) // 10
    # non_n = len(healthy) // 10
    # ds = [depression[i:i + dep_n] for i in range(0, len(depression), dep_n)]
    # hs = [healthy[i:i + non_n] for i in range(0, len(healthy), non_n)]
    # vals = []
    # trains = []
    # for i in range(10):
    #     val = []
    #     train = []
    #     test=[]
    #     val.extend(ds[i])
    #     val.extend(hs[i])
    #     for j in range(10):
    #         if j != i:
    #             train.extend(ds[j])
    #             train.extend(hs[j])
    #     vals.append(val)
    #     trains.append(train)
    dir_list = []
    for i in os.listdir(audio_path):
        dir_list.append(i)
    for i in range(10):
        print("================={}================".format(i))
        # 将数据集按照8:1:1的比例划分为训练集、测试集和验证集
        X_train_val, X_test = train_test_split(dir_list, test_size=0.1, random_state=i)
        X_train, X_val = train_test_split(X_train_val, test_size=1 / 9, random_state=i)
        print("train", X_train)
        print("val", X_val)
        print("test", X_test)
        train_path=f'/usr/data/local/duanyuchi/python_data/multimodal/All_data/c3_s3/train_{i}.txt'
        eval_path= f'/usr/data/local/duanyuchi/python_data/multimodal/All_data/c3_s3/val_{i}.txt'
        test_path= f'/usr/data/local/duanyuchi/python_data/multimodal/All_data/c3_s3/test_{i}.txt'
        with open(train_path, mode='w') as f:
            for name_id in X_train:
                label = '1' if name_id[-5] == '1' else '0'
                for dir_id in sorted(os.listdir(os.path.join(audio_path,name_id))):
                    for wav in sorted(os.listdir(os.path.join(audio_path,name_id,dir_id))):
                        wav_path=os.path.join(audio_path, name_id,dir_id,wav)
                        img_path = os.path.join(video_path, name_id, dir_id, wav[:2]+wav[4:-4] + '.npy')
                        if (os.path.exists(img_path)):
                            f.write(img_path + ';' +wav_path+';'+ label + '\n')
            f.close()
        with open(eval_path, mode='w') as f:
            for name_id in X_val:
                label = '1' if name_id[-5] == '1' else '0'
                for dir_id in sorted(os.listdir(os.path.join(audio_path, name_id))):
                    for wav in sorted(os.listdir(os.path.join(audio_path, name_id, dir_id))):
                        wav_path = os.path.join(audio_path, name_id, dir_id, wav)
                        img_path = os.path.join(video_path, name_id, dir_id, wav[:2]+wav[4:-4] + '.npy')
                        if (os.path.exists(img_path)):
                            f.write(img_path + ';' + wav_path + ';' + label + '\n')
            f.close()

            with open(test_path, mode='w') as f:
                for name_id in X_test:
                    label = '1' if name_id[-5] == '1' else '0'
                    for dir_id in sorted(os.listdir(os.path.join(audio_path, name_id))):
                        for wav in sorted(os.listdir(os.path.join(audio_path, name_id, dir_id))):
                            wav_path = os.path.join(audio_path, name_id, dir_id, wav)
                            img_path = os.path.join(video_path, name_id, dir_id, wav[:2]+wav[4:-4] + '.npy')
                            if (os.path.exists(img_path)):
                                f.write(img_path + ';' + wav_path + ';' + label + '\n')
                f.close()
audio_path='/usr/data/local/duanyuchi/python_data/multimodal/All_data/c3_s3/audio_s_c3_s3/'
video_path='/usr/data/local/duanyuchi/python_data/multimodal/All_data/c3_s3/video_c3_s3/'
seq_npy_txtCreat(audio_path,video_path)