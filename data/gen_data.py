# -*- coding: utf-8 -*-
"""
@author: PengChuan
"""
import os
import numpy as np
import util
from data import data2file
import sound.core as sd_core
import sound.feat as sd_feat
from multiprocessing import Queue, Process
import global_config

# data = {
#     'samplerate': 16000,
#     'windowsize': 320,
#     'hop': 160,
#     'norm_feat': False,
#     'voice_root': 'dataset/TSP.small',
#     'noise_root': 'dataset/Noisex92.one',
#     'dir_max_n': 6,
#     'padding': True,
#     'concat': False,
#     'ext': {
#         'num': 4,
#         'direction': 2,
#         'overlap': True,
#     },
#     'feat': {
#       'feat.clean': [sd_feat.FEAT_MAGNITUDE],
#       'feat.noise': [sd_feat.FEAT_MAGNITUDE],
#       'feat.mix': [sd_feat.FEAT_MAGNITUDE],
#     },
#     'aligin': {
#         # 'too_long': 'append_0',
#         'too_long': 'cut',
#         'too_short': 'append_0',
#         # 'too_short': 'discard',
#     },
#     'train': {
#         'snr': -2,
#     },
#     'eval': {
#         'snr': -2,
#     },
#     'test': {
#         'snr': [-5, -2, 0, 2, 5],
#     },
# }


class Sample(object):
    def __init__(self):
        self.mix_raw = None # mix_raw混合后的时域数据
        self.clean_raw = None
        self.mix_feat = None # 混合数据的特征
        self.mix = None
        self.enhed = None # 增强
        self.enhed_raw = None
        self.clean = None
        self.phase = None # 相位谱
        self.vpath = None
        self.npath = None
        self.stoi = None
        self.pesq = None
        self.ssnr = None


def gen_raw(sample, hop, padding):
    def _gen(mag, phase):
        shape = mag.shape
        if len(shape) == 3:
            mid = shape[1] // 2
            mag = mag[:, mid, :]
        spec = mag * np.exp(1j * phase)
        raw = sd_core.istft(spec, hop)
        if padding:
            raw = raw[hop:-hop] # 做完ISTFT后把补零剪去
        return raw
    if sample.mix_raw is None:
        sample.mix_raw = _gen(sample.mix, sample.phase)
    if sample.clean_raw is None:
        sample.clean_raw = _gen(sample.clean, sample.phase)
    if sample.enhed_raw is None:
        sample.enhed_raw = _gen(sample.enhed, sample.phase)


class Generator(object):
    def __init__(self, config, phase, snr=None):
        self.conf = config
        self.ph = phase # 阶段
        self.vpath = data2file.aligin(config)
        self.sr = config['samplerate']
        if snr is not None:
            self.snr = snr
        elif phase == 'train':
            self.snr = config['train']['snr']
        elif self.ph == 'eval':
            self.snr = config['eval']['snr']
        else:
            self.snr = 0

    def gen_raw(self, sample):
        gen_raw(sample, self.conf['hop'], self.conf['padding'])

    def wav_path(self):
        voice_path = os.path.join(self.vpath, self.ph, 'raw')
        noise_root = os.path.join(self.conf['noise_root'], self.ph)

        def list_wav(data_root):
            file_l = []
            for root, dirs, files in os.walk(data_root):
                for f in files:
                    if f.endswith('.wav'):
                        file_l.append(os.path.join(root, f))
            file_l.sort()
            return file_l
        voice_fl = list_wav(voice_path)
        noise_fl = list_wav(noise_root)

        noise_indx = 0
        cp = []
        for vp in voice_fl:
            cp.append((vp, noise_fl[noise_indx]))
            noise_indx = (noise_indx + 1) % len(noise_fl)

        np.random.shuffle(cp)
        return cp

    def gen_with_path(self, vpath, npath):
        if self.ph == 'train':
            noise_part = 0 # noise
        elif self.ph == 'eval':
            noise_part = 1
        else:
            noise_part = 2

        padding, hop = self.conf['padding'], self.conf['hop'] # hop帧和帧之间的间隔？？？
        v_ftp = self.conf['feat']['feat.clean'] # voice feature type
        m_ftp = self.conf['feat']['feat.mix'] # mix feature type

        mixture, voice, noise = self._mix(vpath, npath, noise_part, self.snr)
        vfeat = self._calc_feat(voice, v_ftp, padding, hop)
        # nfeat = self._calc_feat(noise, n_ftp, ext_d, overlap, 0, padding, hop)
        mfeat = self._calc_feat(mixture, m_ftp, padding, hop)
        m_mag = self._calc_feat(mixture, sd_feat.FEAT_MAGNITUDE, padding, hop)
        # TODO BUG: 多进程环境下，此函数_calc_feat阻塞，具体原因不明。
        # 逐步排查，定位到_calc_feat#compute_feat#sd_feat.compute_feat#np.angle#arctan2
        phase = self._calc_feat(mixture, sd_feat.FEAT_PHASE, padding, hop)

        sp = Sample()
        sp.mix_feat, sp.mix, sp.clean, sp.phase = mfeat, m_mag, vfeat, phase
        sp.vpath, sp.npath = vpath, npath
        sp.clean_raw, sp.mix_raw = voice, mixture
        return sp

    def asyn_gen(self, n_proc=-1, buffersize=8):
        # TODO BUG: 此函数在n_proc大于1且在windows平台运行，可能产生错误。
        # multiprocssing的子进程生成过程在Windows与Unix平台上不同，
        # windows必须生成全新的进程，从程序的起点开始运行， 而unix在直接调用处fork()。
        # 参考https://docs.python.org/3/library/multiprocessing.html#the-spawn-and-forkserver-start-methods
        # 需要确保main module被解释器安全import，避免意外的副作用。
        # 这里不能用multiprocssing.Pool管理进程，会产生AttributeError，原因不明

        if n_proc <= 0:
            n_proc = global_config.NUM_PROCESS

        if n_proc == 1:
            for path in self.wav_path():
                yield self.gen_with_path(*path)
            return
        n_workers = n_proc - 1

        q_task = Queue()
        q_data = Queue(buffersize)
        for path in self.wav_path():
            q_task.put(path)
        for i in range(n_workers):
            p_gen = Process(target=_gen, args=(self, q_task, q_data))
            q_task.put(None)
            p_gen.start()
        n_finished = 0
        while True:
            value = q_data.get()
            if value is None:
                n_finished += 1
                if n_finished >= n_workers:
                    break
            else:
                yield value

    def _mix(self, voice_f, noise_f, noise_part, snr):
        # TODO BUG: 调用librosa读取音频可能会造成阻塞，具体原因不明。
        # 这里换用soundfile库，能否解决问题待观察
        voice, sr = util.wav_read(voice_f, self.sr)
        noise, sr = util.wav_read(noise_f, self.sr)
        if len(noise) < len(voice) * 3:
            noise = np.tile(noise, int(np.ceil(len(voice) * 3 / len(noise))))
        one_third = int(len(noise) / 3)
        noise = noise[one_third * noise_part:one_third * noise_part+len(voice)]

        n_2 = np.sum(noise ** 2)
        if n_2 > 0.:
            a = np.sqrt(np.sum(voice ** 2) / (n_2 * 10 ** (snr / 10)))
            noise = a * noise
        mixture = voice + noise
        return mixture, voice, noise

    def _calc_feat(self, wav, feat, padding, hop):
        if padding:
            wav = np.pad(wav, (hop, hop), 'constant', constant_values=(0, 0))
        trans = compute_feat(wav, feat, self.conf)
        # trans = extend(trans, ext_n, ext_d, overlap)
        return trans


def _gen(generator, q_task, q_data):
    # multiprocessing使用pickle序列化为多进程传递参数，而pickle要求运行函数必须为非局部函数。
    # 如果传入局部函数，在windows下，会产生AttributeError，然而在ubuntu下没有此错误。
    while True:
        # print("gen: %d" % q_data.qsize())
        value = q_task.get()
        if value is None:
            q_data.put(None)
            break
        d = generator.gen_with_path(*value)
        q_data.put(d)


def compute_feat(wav, feat, config):
    return sd_feat.compute_feat(wav, feat, config['samplerate'],
                                config['windowsize'], config['hop'], norm=config['norm_feat'])


def adj_extend(sig, num, direction, overlap):
    extension = []
    step = 1 if overlap else (num * direction + 1)
    for i in range(0, len(sig), step):  # 帧数
        temp = []
        start = i if direction <= 1 else i - num
        for k in range(start, i + num + 1):
            if k < 0 or k >= len(sig):
                temp.append(np.zeros(len(sig[i])))
            else:
                temp.append(sig[k])
        extension.append(np.array(temp))
    return np.array(extension)


# import tensorflow as tf
# generator = Generator(data, 'train', 4, 4)
# ds = tf.data.Dataset.from_generator(generator.gen, (tf.float32, tf.float32, tf.float32, tf.float32))
# g_in, d_in, voice = ds.make_one_shot_iterator().get_next()
#
# with tf.Session() as sess:
#     # sess.run(value)  # (1, array([1]))
#     # sess.run(value)  # (2, array([1, 1]))
#     print(sess.run(g_in).shape)
#     # g_in, d_in, voice =sess.run(value)
#     # print(g_in.shape)
#     # print(d_in.shape)
#     # print(voice.shape)