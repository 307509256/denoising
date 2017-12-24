# -*- coding: utf-8 -*-

"""
各种工具方法，服务于各处理模块
"""

import os
import shutil
import nnresample
import numpy as np
import soundfile
import platform


def mkdir_p(dir_path):
    '''
    mkdri -p，创建目录，不存在则创建，如必要也创建上级目录
    :param dir_path: 目录路径
    '''
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def rm_r(path):
    '''
    删除操作
    '''
    if os.path.isdir(path):
        shutil.rmtree(path)
    elif os.path.exists(path):
        os.remove(path)


def standardize(vec):
    '''
    标准化，使向量均值为0,方差为1
    '''
    mean = np.mean(vec)
    vec -= mean
    var = np.sum(vec * vec)
    vec /= np.sqrt(var)
    return vec


def wav_read(path, samplerate=None):
    y, sr = soundfile.read(path)
    if samplerate is not None and samplerate != sr:
        y = resample(y, sr, samplerate)
        sr = samplerate
    return y, sr


def wav_write(path, wav, sr, norm=False):
    if norm:
        wav = wav / np.max(np.abs(wav)) * 0.99999 # 所有的乘以0.9999，<1
    soundfile.write(path, wav, sr)


def resample(wav, old_sr, new_sr):
    return nnresample.resample(wav, new_sr, old_sr)


def os_windows():
    return 'Windows' in platform.system()


def os_linux():
    return 'Linux' in platform.system()
