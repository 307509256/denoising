# -*- coding: utf-8 -*-
"""
音频处理模块。包含傅里叶变换相关处理。
@author: PengChuan
"""


import numpy as np
import scipy.signal

# 决定音频首尾是否各加一“镜像”帧
CENTER_DEFAULT = False

def stft(frames, n_fft=None, window='hann', half=True):
    # 短时傅里叶变换
    w = scipy.signal.get_window(window, frames.shape[1])
    spec = np.fft.fft(frames * w, n=n_fft).conj()
    if half: # 防止数据冗余
        spec = spec[:, :frames.shape[1] // 2 + 1]
    return spec


def istft(spec, frame_shift, window='hann', half=True, center=CENTER_DEFAULT):
    # 短时傅里叶逆变换
    if half:
        spec = np.concatenate((spec.conj(), spec[:, -2:0:-1]), axis=1)

    frame_size = spec.shape[1]
    sig_len = frame_size + (spec.shape[0] - 1) * frame_shift
    sig = np.zeros(sig_len)

    w = scipy.signal.get_window(window, frame_size)
    window_sum = np.zeros(sig_len)
    window_square = w * w

    for i in range(len(spec)):
        hop = i * frame_shift
        wav = np.fft.ifft(spec[i], frame_size).real
        sig[hop:hop + frame_size] += wav * w

        window_sum[hop:hop + frame_size] += window_square

    nonzero_indices = window_sum > np.finfo(w.dtype).tiny
    sig[nonzero_indices] /= window_sum[nonzero_indices]

    if center:
        sig = invert_center_wav(sig, frame_size)

    return sig  #signal


def frame(wav, frame_size, frame_shift, pad=False, center=CENTER_DEFAULT):
    '''
    对时域信号分帧
    :param wav: 原始wav格式的音频文件
    :param frame_size:  每帧大小。用于stft的分帧，帧大小应该为偶数
    :param frame_shift:
    :param pad: 如果采样点有剩余，选择True补0成一帧，或False丢弃
    :param center: 首尾各加一“镜像”帧，输入之前音频首尾补0，突兀，。。。。
    :return:
    '''
    if pad:
        wav, _ = int_times_frame(wav, frame_size, frame_shift, pad)
    if center:
        wav = center_wav(wav, frame_size)

    frames = []
    for i in range(0, len(wav) - frame_size + 1, frame_shift):
        frames.append(wav[i:i + frame_size])
    return np.array(frames)


def iframe(frames, frame_size, center=CENTER_DEFAULT):
    # 帧----->wav格式   inverse
    wav = []
    for fr in frames:
        wav.extend(fr)
    if center:
        wav = invert_center_wav(wav, frame_size)
    return wav


def center_wav(wav, n):
    return np.pad(wav, n, mode='reflect')


def invert_center_wav(wav, n):
    return wav[n:-n]


def int_times_frame(wav, frame_size, frame_shift, pad=False):
    '''
    裁剪或补全时域信号，使之在分帧时不会剩余采样点
    :param wav:
    :param frame_size:
    :param frame_shift:
    :param pad: True补全，False裁剪
    :return: 处理后的信号及裁剪或补全的采样点数量
    '''
    surplus = (len(wav) - frame_size) % frame_shift  # 剩余的采样点数量
    if surplus == 0:
        return wav[:], 0

    if pad:
        change = frame_shift - surplus
        # 在信号的末尾补0
        wav = np.pad(wav, (0, change), 'constant', constant_values=(0, 0))
    else:
        change = surplus
        wav = wav[:len(wav) - surplus]

    return wav, change


def get_mag(wav, frame_size, frame_shift):
    '''
    计算音频的幅度谱
    :param wav: 音频时域信号
    :param frame_size: 帧大小
    :param frame_shift: 帧移
    :return: 幅度谱，二阶array，shape(帧数目，每帧傅里叶系数数目)
    '''
    frames = frame(wav, frame_size, frame_shift)
    feature = stft(frames)
    return np.absolute(feature)


def get_mag_dim(frame_size):
    # 为什么要除以二加1，因为幅度谱关于中心对称吗？冗余
    return frame_size // 2 + 1