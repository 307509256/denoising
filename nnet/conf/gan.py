# -*- coding: utf-8 -*-
"""
@author: PengChuan
"""

import tensorflow as tf
import sound.feat as sd_feat

data = {
    'samplerate': 16000,
    'windowsize': 320,
    'hop': 160,
    'norm_feat': False,
    'voice_root': 'dataset/TSP.one/',
    # 'voice_root': 'dataset/TIMIT+Noisex92/TIMIT',
    'noise_root': 'dataset/Noisex92.all',
    'dir_max_n': 6,
    'padding': True,
    'concat': False,
    'ext': {
        'num': 4,
        'direction': 2,
        'overlap': True,
    },
    'feat': {
      'feat.clean': [sd_feat.FEAT_MAGNITUDE],
      'feat.noise': [sd_feat.FEAT_MAGNITUDE],
      'feat.mix': [sd_feat.FEAT_MAGNITUDE],
    },
    'aligin': {
        # 'too_long': 'append_0',
        'too_long': 'cut',
        'too_short': 'append_0',
        # 'too_short': 'discard',
    },
    'train': {
        'snr': -2,
    },
    'eval': {
        'snr': -2,
    },
    'test': {
        'snr': [-5, -2, 0, 2, 5],
    },
}

MODEL_PATH = 'model/gan'
CONDITIONAL_GAN = False
LS_GAN = False
GRADIENT_PENALTY = True
G_L1_REGULAR = 0
G_LR = 5e-5     # G的学习速率
D_LR = 5e-5   # D的学习速率
G_EXTEND = 4   # G输入的每帧前/后扩展帧数目
D_EXTEND = 4   # D输入的每帧前/后扩展帧数目
GP_LAMBDA = 10.

# G_L2_REGULAR = 1e-2
G_L2_REGULAR = 0
# D_L2_REGULAR = 8e-2
D_L2_REGULAR = 0
OPT_ADAM = False
# 模型训练次数
MAX_ITERATION = 100
# 每次G迭代后，D迭代次数。如果小于0，表示每迭代-D_ITER次G，迭代一次D
D_ITER = 4

INPUT_DIM = sd_feat.get_frame_dim(data['feat']['feat.mix'], data['windowsize'])
MAG_DIM = sd_feat.get_frame_dim(sd_feat.FEAT_MAGNITUDE, data['windowsize'])
OUTPUT_DIM = MAG_DIM

G_NN_STRUCT = [
    {'type': 'conv', 'n_out': 8, 'ksize': 4, 'stride': 1, 'fun': tf.nn.relu, 'norm_fn': None},
    # {'type': 'dropout', 'keep': 0.8},
    {'type': 'full', 'n_out': 1024, 'fun': tf.nn.relu, 'norm_fn': None},
    # {'type': 'dropout', 'keep': 0.8},
    {'type': 'full', 'n_out': 1024, 'fun': tf.nn.relu, 'norm_fn': None},
    # {'type': 'dropout', 'keep': 0.8},
    {'type': 'full', 'n_out': 1024, 'fun': tf.nn.relu, 'norm_fn': None},
    # {'type': 'dropout', 'keep': 0.8},
    {'type': 'full', 'n_out': OUTPUT_DIM, 'fun': tf.nn.relu},
]

D_NN_STRUCT = [
    {'type': 'conv', 'n_out': 16, 'ksize': 4, 'stride': 2, 'fun': tf.nn.relu, 'norm_fn': None},
    # {'type': 'dropout', 'keep': 0.95},
    # {'type': 'pool', 'ksize': [1, 1, 5, 1], 'stride': [1, 1, 5, 1]},
    {'type': 'conv', 'n_out': 16, 'ksize': 4, 'stride': 2, 'fun': tf.nn.relu, 'norm_fn': None},
    # {'type': 'dropout', 'keep': 0.95},
    {'type': 'conv', 'n_out': 16, 'ksize': 4, 'stride': 2, 'fun': tf.nn.relu, 'norm_fn': None},
    # {'type': 'dropout', 'keep': 0.95},
    {'type': 'conv', 'n_out': 16, 'ksize': 4, 'stride': 2, 'fun': tf.nn.relu, 'norm_fn': None},
    # {'type': 'dropout', 'keep': 0.95},
    # {'type': 'full', 'n_out': 128, 'fun': tf.nn.relu},
    {'type': 'full', 'n_out': 1, 'fun': None},
]
