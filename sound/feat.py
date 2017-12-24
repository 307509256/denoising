# -*- coding: utf-8 -*-
"""
@author: PengChuan
"""

import numpy as np
from sound import core
from sound import gammatone, mel_spec, pncc


FEAT_LOG_MEL = "LOG-MEL"  # log-mel filter bank 滤波器组
FEAT_MAGNITUDE = "MAG"  # magnitude spectrum 幅度谱
FEAT_PNCC = 'PNCC'  # Power-Normalized Cepstral Coefficients 归一化功率倒谱系数pncc对噪声容忍。。。
FEAT_GF = 'GF'  # Gammatone Feature
FEAT_GFCC = 'GFCC'  # Gammatone Frequency Cepstral Coefficients 基于 Gammatone 滤波器的听觉模型倒谱特征参数
FEAT_WAV = 'WAV'

FEAT_SPECTRUM = "spectrum" # 语谱图 复数
FEAT_PHASE = "phase" # 相位

MEL_COEF_N = 40# mel 频率的系数的个数，取多少系数！查一下
GAMMATONE_FILTER_N = gammatone.N_FILTERS_DEFAULT


def standardize(vec):
    '''
    标准化，使向量均值为0,方差为1
    '''
    mean = np.mean(vec)
    vec -= mean
    var = np.sum(vec * vec)
    vec /= np.sqrt(var)
    return vec


def compute_feat(wav, feat, samplerate, windowsize, hop_point, norm=False, center=False):
    '''
    从时域信号中计算出feat特征
    :param feat: 特征类型
    '''
    if type(feat) == list or type(feat) == tuple:
        feature = [compute_feat(wav, f, samplerate, windowsize, hop_point,
                                norm=norm, center=center) for f in feat]
        return np.concatenate(feature, axis=1)

    if feat == FEAT_MAGNITUDE:
        frames = core.frame(wav, windowsize, hop_point, center=center)
        feature = core.stft(frames)
        feature = np.absolute(feature)
    elif feat == FEAT_LOG_MEL:
        feature = mel_spec.melspectrogram(wav, samplerate, windowsize, hop_point, MEL_COEF_N)
        feature = mel_spec.power_to_db(feature)
    elif feat == FEAT_PNCC:
        feature = pncc.calc_pncc(wav, samplerate, windowsize, hop_point)
    elif feat == FEAT_GF:
        feature = gammatone.gammatonegram(wav, samplerate, windowsize,
                                          hop_point, GAMMATONE_FILTER_N)
    elif feat == FEAT_GFCC:
        feature = gammatone.gammatonegram(wav, samplerate, windowsize,
                                          hop_point, GAMMATONE_FILTER_N)
        feature = gammatone.gtm2gfcc(feature, dct_stop=31)
    elif feat == FEAT_SPECTRUM:
        frames = core.frame(wav, windowsize, hop_point, center=center)
        feature = core.stft(frames)
    elif feat == FEAT_PHASE:
        frames = core.frame(wav, windowsize, hop_point, center=center)
        feature = core.stft(frames)
        feature = np.angle(feature)
    elif feat == FEAT_WAV:  # raw
        feature = core.frame(wav, windowsize, hop_point, center=center)
    else:
        raise ValueError("feature %s not support" % feat)

    if norm:
        feature = [standardize(f) for f in feature]

    return np.array(feature)


def get_frame_dim(feat, windowsize):
    # 获取帧的维度
    if type(feat) == list or type(feat) == tuple:
        dim = 0
        for f in feat:
            dim += get_frame_dim(f, windowsize)
        return dim

    if feat == FEAT_MAGNITUDE or feat == FEAT_SPECTRUM\
            or feat == FEAT_PHASE:
        dim = core.get_mag_dim(windowsize)
    elif feat == FEAT_LOG_MEL:
        dim = MEL_COEF_N
    elif feat == FEAT_PNCC:
        dim = pncc.DCT_NUM
    elif feat == FEAT_GF:
        dim = GAMMATONE_FILTER_N
    elif feat == FEAT_GFCC:
        dim = gammatone.GFCC_NUM_DEFAULT
    elif feat == FEAT_WAV:  # raw
        dim = windowsize
    else:
        raise ValueError("feature %s not support" % feat)

    return dim
