# 这个代码主要是使用面部标志点去当作特征送入网络，也算是一种尝试，对视频特征，每一秒平均成一个136维度的向量
import math
import os
import pandas as pd
import numpy as np

#这个代码写的太丑陋了，可以用一个字典去存储的，
def pro(csvm, root_out):
    data = pd.read_csv(csvm)
    flag = math.ceil(data.iloc[-1, 2])
    rows = data.shape[0]
    right = 0
    sums = []
    for i in range(rows):
        if (data.iloc[i, 2] > right):
            sums.append(i - 1)
            right += 1
    sums.append(data.shape[0])
    print(sums)
    results = []
    for j in range(len(sums) - 1):
        ans = data.iloc[sums[j]:sums[j + 1], 299:].mean(axis=0)
        ans = np.array(ans)
        results.append(ans)
    np.save(os.path.join(root_out ,root_out[-2:]+'.npy'), np.array(np.array(results)))


root_img = '/usr/data/local/duanyuchi/python_data/multimodal/All_data/video2img/'
root_out = '/usr/data/local/duanyuchi/python_data/multimodal/All_data/video_136/'
for papers in sorted(os.listdir(root_img)):
    for ids in os.listdir(os.path.join(root_img, papers)):
        for dirs in os.listdir(os.path.join(root_img, papers, ids)):
            if (dirs.endswith('.csv')):
                csv = os.path.join(root_img, papers, ids, dirs)
                print(csv)
                os.makedirs(os.path.join(root_out,papers,ids))
                pro(csv, os.path.join(root_out, papers, ids))
