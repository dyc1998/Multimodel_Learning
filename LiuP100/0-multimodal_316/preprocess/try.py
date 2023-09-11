import numpy as np
import os

# 加载npy文件
video_data = np.load('/usr/data/local/lyt_data/dataset13/valid/203_2_aligned/video.npy')
# video_data = np.load('/usr/data/local/duanyuchi/python_data/multimodal/ThreeOnly18/video_c3s_s1s/1-001-06010010/18-1/18-1_000.npy')

# 查看视频数据的形状
print("视频数据的形状:", video_data.shape)

# 查看视频数据的内容
print("视频数据的内容:", video_data)
# 获取当前脚本的路径
script_path = os.path.abspath(__file__)

# 打印当前脚本的路径
print("当前脚本的路径:", script_path)