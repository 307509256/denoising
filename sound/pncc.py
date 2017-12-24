# -*- coding: utf-8 -*-

import scipy.signal
import sound.gammatone as gmt
import sound.core as basic
from scipy.fftpack import dct
import numpy as np
# import scipy.io

_SAMPLE_RATE = 16000
_LOW_FREQ = 200

_PRE_EMPHSIS = True  # 预加重
_CALC_MEDIUM_DURATION = True  # 计算中等时长功率谱
_POWER_NONLINEARITY = True  # True power nonlinearity, False logarithmic nonlinearity

_DELTA = 0.01
_POWER_COEFF = 1. / 15

_FFT_SIZE = 1024
_FILTER_NUM = 40

_NORM_POWER = 1e15
_EPS = np.finfo(float).eps

_SMTH_FRM, _SMTH_FLT = 2, 4  # 平滑处理的帧数目和通道数目

DCT_NUM = 13    # DCT(Discrete Cosine Transform,离散余弦变换)系数数目，也是最后PNCC系数数目

def calc_pncc(sig, sr=_SAMPLE_RATE, frame_size=410, frame_shift=160):
    '''
    计算Power-Normalized Cepstral Coefficients (PNCC)功率归一化倒谱系数
    :param sig: 原始时域信号
    :param sr: 采样频率，默认16kHz，若不同，会首先重采样
    :param frame_size: 每帧采样点数，默认对应25.6ms
    :param frame_shift: 每帧移动采样点数，默认对应10ms
    :return: PNCC特征
    '''
    # if sr != _SAMPLE_RATE:
    #     sig = librosa.resample(sig, sr, _SAMPLE_RATE)

    # STFT
    frames = basic.frame(sig, frame_size, frame_shift)
    frames = basic.stft(frames, n_fft=_FFT_SIZE, window='hamming', half=False)
    frames = frames[:, :_FFT_SIZE // 2]
    frames = np.abs(frames)

    num_frame = len(frames)

    if _PRE_EMPHSIS:
        # Pre - emphasis using H(z) = 1 - 0.97 z^-1
        sig[1:] -= 0.97 * sig[:-1]

    # Obtaning the gammatone coefficient.
    aad_H = _calc_gammatone_filter_response(_FILTER_NUM, _FFT_SIZE, sr)
    aad_H = np.abs(_normalize_gain(aad_H))

    # x = sig[:_FRAME_SIZE]
    # w = scipy.signal.get_window('hamming', _FRAME_SIZE)
    # c = w * x
    # scipy.io.savemat('sig.mat', {'sig':c})
    # X = np.fft.fft(c, _FFT_SIZE)

    # Obtaining the short-time Power
    aad_P, ad_sum_P = [], []
    aad_HT = aad_H.T
    for frame in frames:
        aad_P.append(np.sum((aad_HT * frame) ** 2, axis=1))
        ad_sum_P.append(np.sum(aad_P[-1]))
    aad_P = np.array(aad_P).T

    # window = scipy.signal.get_window('hamming', _FRAME_SIZE)
    # aad_P = np.zeros((_FILTER_NUM, num_frame))
    # ad_sum_P = []
    # fi = 0
    # for i in range(0, len(sig) - _FRAME_SIZE + 1, _FRAME_SHIFT):
    #     ad_x_st = sig[i:i+_FRAME_SIZE]
    #     ad_x_st = ad_x_st * window
    #     adSpec = np.fft.fft(ad_x_st, _FFT_SIZE)
    #     ad_X = np.abs(adSpec[:_FFT_SIZE//2])
    #     for j in range(_FILTER_NUM):
    #         aad_P[j, fi] = np.sum((ad_X * aad_H[:, j])**2)
    #     ad_sum_P.append(np.sum(aad_P[:, fi]))
    #     fi += 1

    # Peak Power Normalization Using 95 % percentile
    ad_sum_P.sort()
    max_p = ad_sum_P[np.round(0.95 * len(ad_sum_P)).astype(int) - 1]
    aad_P = aad_P / max_p * _NORM_POWER
    # scipy.io.savemat('aad_P.mat', {'aad_P_tmp': aad_P})

    if _CALC_MEDIUM_DURATION:
        # Medium-duration power calculation
        aad_Q = []
        for i in range(_FILTER_NUM):
            q = []
            for j in range(num_frame):  # frame number
                q.append(np.mean(
                    aad_P[i, max(0, j - _SMTH_FRM):min(num_frame, j + _SMTH_FRM + 1)]))
            aad_Q.append(q)
        aad_Q = np.array(aad_Q)

        aad_w = []
        for i in range(_FILTER_NUM):
            aad_tildeQ = _power_bias_sub(aad_Q[i, :], _DELTA)
            aad_w.append(_max(aad_tildeQ, _EPS) / _max(aad_Q[i, :], _EPS))
        aad_w = np.array(aad_w)

        # Weight smoothing aross channels
        aad_w_Smooth = np.zeros(aad_Q.shape)
        for i in range(_FILTER_NUM):
            for j in range(num_frame):
                aad_w_Smooth[i, j] = np.mean(
                    aad_w[max(i - _SMTH_FLT, 0):min(i + _SMTH_FLT + 1, _FILTER_NUM), j])

        aad_P *= aad_w_Smooth
        # aad_P = aad_P[:, _SMTH_FRM:aad_P.shape[1] - _SMTH_FRM - 1]

        # Apply the nonlinearity
        if _POWER_NONLINEARITY:
            aadSpec = aad_P ** _POWER_COEFF
        else:
            aadSpec = np.log(aad_P + _EPS)

        # DCT
        aadDCT = dct(aadSpec, norm='ortho', axis=0)
        aadDCT = aadDCT[:DCT_NUM, :]

        # CMN
        for i in range(DCT_NUM):
            aadDCT[i, :] -= np.mean(aadDCT[i, :])

        return aadDCT.T


def _max(arr, num):
    return np.array([max(a, num) for a in arr])


def _power_bias_sub(ad_Q, delta):
    ad_B = [_NORM_POWER / (10 ** (j / 10) + 1) for j in range(70, 9, -1)]
    ad_B.insert(0, 0)
    tildeGTemp = 0
    ad_tildeQSave = ad_Q

    for d_B in ad_B:
        aiIndex = np.where(ad_Q > d_B)
        if len(aiIndex) <= 0:
            break

        posMean = np.mean(ad_Q[aiIndex] - d_B)
        # print(d_B + delta * posMean)
        aiIndex = np.where(ad_Q > (d_B + delta * posMean))
        if len(aiIndex) <= 0:
            break

        cf = np.mean(ad_Q[aiIndex] - d_B) * delta
        ad_tildeQ = _max(ad_Q - d_B, cf)
        adData = ad_tildeQ[aiIndex]

        tildeG = np.log(np.mean(adData)) - np.mean(np.log(adData))
        if tildeG > tildeGTemp:
            ad_tildeQSave = ad_tildeQ
            tildeGTemp = tildeG

    return ad_tildeQSave


def _calc_gammatone_filter_response(channel_num, fft_size, samplerate=_SAMPLE_RATE):
    fcoef = gmt.make_erb_filters(samplerate, channel_num, _LOW_FREQ)
    A0 = fcoef[:, 0]
    A11 = fcoef[:, 1]
    A12 = fcoef[:, 2]
    A13 = fcoef[:, 3]
    A14 = fcoef[:, 4]
    A2 = fcoef[:, 5]
    B0 = fcoef[:, 6]
    B1 = fcoef[:, 7]
    B2 = fcoef[:, 8]
    gain = fcoef[:, 9]

    H = np.zeros(shape=(fft_size // 2, len(gain)), dtype=np.complex128)
    for chan in range(len(gain)):
        HDen = [B0[chan], B1[chan], B2[chan]]

        H1Num = [A0[chan] / gain[chan], A11[chan] / gain[chan],
                 A2[chan] / gain[chan]]
        W, H1 = scipy.signal.freqz(H1Num, HDen, fft_size // 2)

        H2Num = [A0[chan], A12[chan], A2[chan]]
        W, H2 = scipy.signal.freqz(H2Num, HDen, fft_size // 2)

        H3Num = [A0[chan], A13[chan], A2[chan]]
        W, H3 = scipy.signal.freqz(H3Num, HDen, fft_size // 2)

        H4Num = [A0[chan], A14[chan], A2[chan]]
        W, H4 = scipy.signal.freqz(H4Num, HDen, fft_size // 2)

        H[:, len(gain) - 1 - chan] = H1 * H2 * H3 * H4

    return H


def _normalize_gain(aad_H):
    ad_w = np.abs(aad_H * aad_H)
    ad_w = np.sqrt(np.sum(ad_w, axis=0))
    ad_w /= ad_w[0]

    aad_H /= ad_w
    return aad_H


# from scipy.fftpack import dct
#
# x = np.array([2, 3])
# print(dct(x))
# ad_B = [_NORM_POWER / (10 ** (j / 10) + 1) for j in range(70, 9, -1)]
# ad_B.insert(0, 0)
# print(ad_B)
# import numpy as np
#
# wav, fs = librosa.load('data/TSP/test/FA06_01.wav', sr=None)
# scipy.io.savemat('sig.mat', {'sig': wav})
# # wav = np.array([3.,4.,1.])
# feat = calc_pncc(wav, fs)
# print(feat)
# # print(gmt.__file__)
# # _calc_gammatone_filter_response(40, 1024)
