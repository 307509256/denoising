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
        'num': 0,
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

MODEL_PATH = 'model/rnn'
INPUT_DIM = sd_feat.get_frame_dim(data['feat']['feat.mix'], data['windowsize'])
MAG_DIM = sd_feat.get_frame_dim(sd_feat.FEAT_MAGNITUDE, data['windowsize'])
OUTPUT_DIM = MAG_DIM

# 神经网络输入，每帧前/后扩展数
EXTEND_NUM = 0
L2_REGULAR = 0
CLIP_WEIGHT = True

_g_out_dim = MAG_DIM * 2
NN_STRUCT = [
    # {'type': 'conv', 'n_out': 8, 'ksize': 4, 'stride': 1, 'fun': tf.nn.relu, 'norm_fn': None},
    # {'type': 'dropout', 'keep': 0.7},
    {'type': 'rnn', 'n_out': 1024, 'fun': tf.nn.relu},
    {'type': 'full', 'n_out': 1024, 'fun': tf.nn.relu, 'norm_fn': None},
    {'type': 'full', 'n_out': 1024, 'fun': tf.nn.relu, 'norm_fn': None},
    # {'type': 'dropout', 'keep': 0.7},
    {'type': 'full', 'n_out': 1024, 'fun': tf.nn.relu, 'norm_fn': None},
    # {'type': 'dropout', 'keep': 0.7},
    {'type': 'full', 'n_out': 1024, 'fun': tf.nn.relu, 'norm_fn': None},
    # {'type': 'dropout', 'keep': 0.7},
    {'type': 'full', 'n_out': _g_out_dim, 'fun': tf.nn.relu},
]

# 模型训练次数
MAX_ITERATION = 100
LEARNING_RATE = 5e-5
