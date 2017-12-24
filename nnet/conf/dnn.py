# -*- coding: utf-8 -*-
"""
@author: PengChuan
"""

import tensorflow as tf
import sound.feat as sd_feat

data = {
    'samplerate': 16000, # 采样率16000
    'windowsize': 320, # 窗长320
    'hop': 160, # 帧移，一般是窗长的一半
    'norm_feat': False, # 是否对feat正则化
    'voice_root': 'dataset/TSP.one/',
    'noise_root': 'dataset/Noisex92.all',
    'dir_max_n': 6, # 文件中数量最多选6个
    'padding': True, # 不足2^N，自动补零
    'concat': False, # 是否连接下一个样本。。。。
    'ext': {
        'num': 4, # 延拓采样点数是 4
        'direction': 2, # 延拓方向，双向，所以是2，如果是1，则是向右，0不扩展
        'overlap': True, # 重叠 True
    },
    'feat': {
      'feat.clean': [sd_feat.FEAT_MAGNITUDE], # 特征采用幅度谱，也可以选用其他谱，具体看feat.py文件
      'feat.noise': [sd_feat.FEAT_MAGNITUDE],
      'feat.mix': [sd_feat.FEAT_MAGNITUDE],
    },
    'aligin': { #
        # 'too_long': 'append_0',
        'too_long': 'cut',
        'too_short': 'append_0',
        # 'too_short': 'discard',
    },
    'train': {
        'snr': -2, # 信噪比-2db
    },
    'eval': {
        'snr': -2,
    },
    'test': {
        'snr': [-5, -2, 0, 2, 5],
    },
}

MODEL_PATH = 'model/dnn' # 生成模型的路径
INPUT_DIM = sd_feat.get_frame_dim(data['feat']['feat.mix'], data['windowsize']) # 混合音频的输入维度
MAG_DIM = sd_feat.get_frame_dim(sd_feat.FEAT_MAGNITUDE, data['windowsize']) # 幅度谱维度
OUTPUT_DIM = MAG_DIM # 输出维度

# 神经网络输入，每帧前/后扩展数
EXTEND_NUM = data['ext']['num']
L2_REGULAR = 0 # L2范数的系数，L2范数=0，表示不用规范化
CLIP_WEIGHT = True # DNN 防止梯度爆炸进行限制

_g_out_dim = MAG_DIM * 2 # 黄培森 一段是干净语音一段是噪声，所以要乘以二
NN_STRUCT = [
    # {'type': 'conv', 'n_out': 8, 'ksize': 4, 'stride': 1, 'fun': tf.nn.relu, 'norm_fn': None},
    # {'type': 'dropout', 'keep': 0.7},
    {'type': 'full', 'n_out': 1024, 'fun': tf.nn.relu, 'norm_fn': None}, # 第一个hide 层
    {'type': 'full', 'n_out': 1024, 'fun': tf.nn.relu, 'norm_fn': None}, # 第二个hide 层
    # {'type': 'dropout', 'keep': 0.7},
    {'type': 'full', 'n_out': 1024, 'fun': tf.nn.relu, 'norm_fn': None}, # 。。。。
    # {'type': 'dropout', 'keep': 0.7},
    {'type': 'full', 'n_out': 1024, 'fun': tf.nn.relu, 'norm_fn': None},
    # {'type': 'dropout', 'keep': 0.7},
    {'type': 'full', 'n_out': _g_out_dim, 'fun': tf.nn.relu}, # 输出层
]

# 模型训练次数
MAX_ITERATION = 100
LEARNING_RATE = 5e-5
