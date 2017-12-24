# -*- coding: utf-8 -*-
"""
@author: PengChuan
"""

import os
import util
import sound.core as sd_core
import sound.feat as sd_feat
import numpy as np
import json
import hashlib
import librosa
import shutil
import tensorflow as tf

data_root = './dataset'
check_aligin_file, check_mix_file, check_feat_file = '.aligin', '.mix', '.feat'
check_tfrecord_file = '.tfrecord'


# config = {
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
#         'num.clean': 4,
#         'num.noise': 0,
#         'num.mix': 4,
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


def check_trans_exists(config, check_file):
    hashcode = calc_hashcode(config)
    path = os.path.join(data_root, hashcode, check_file)
    return os.path.exists(path), hashcode


def calc_hashcode(config):
    '''
    预处理！！！！！！！！！！！！
    :param config: 字典
    :return:
    '''
    confs = json.dumps(config, sort_keys=True)
    return hashlib.md5(confs.encode('utf-8')).hexdigest()


def get_data_root(config):
    hashcode = calc_hashcode(config)
    return os.path.join(data_root, hashcode)


def compute_feat(wav, feat, config):
    return sd_feat.compute_feat(wav, feat, config['samplerate'],
                                config['windowsize'], config['hop'], norm=config['norm_feat'])


def istft(spectrum, hop, padding):
    '''
    短时傅里叶逆变换
    :param spectrum:
    :param hop:
    :param padding:
    :return:
    '''
    wav = sd_core.istft(spectrum, hop, center=False)
    if padding > 0:
        wav = wav[padding:-padding]
    return wav


# directory tree
# dataset---|---train|------raw|---0|---xxx+xxx.mix.wav
#           |        |         |    |---xxx.voice.wav
#           |        |         |    |---xxx+xxx.noise.wav
#           |        |         |    |---...
#           |        |         |---...
#           |        |
#           |        |-----feat|---0|---xxx+xxx.mix
#           |                  |    |-----xxx.voice
#           |                  |    |---xxx+xxx.noise
#           |                  |    |---...
#           |                  |---...
#           |
#           |----eval|------raw|---...
#           |        |
#           |        |-----feat|---...
#           |
#           |----test|------raw|---0|---xxx1.voice.wav
#                    |         |    |---xxx2.voice.wav
#                    |         |    |---...
#                    |         |---...
#                    |
#                    |--5|--raw|---...
#                    |   |
#                    |   |-feat|---...
#                    |
#                    |--0|---...


def list_all_wav(dir):
    wavs = []
    for root, dirs, files in os.walk(dir):
        wavs.extend([os.path.join(root, f) for f in files if f.endswith('.wav')])
    return wavs


def aligin(config):
    '''
    数据的调整
        补全/切掉/把另一段音频补过来，帧正好是模型输入的整数倍
    :param config:
    :return:
    '''
    existed, code = check_trans_exists(config, check_aligin_file)
    output_root = os.path.join(data_root, code)
    if existed:
        return output_root
    util.mkdir_p(output_root)
    with open(os.path.join(output_root, 'config.json'), 'w') as outfile:
        json.dump(config, outfile, indent=4, sort_keys=True)

    ext_conf = config['ext']
    ali_conf = config['aligin']
    samplerate, padding = config['samplerate'], config['padding']
    file_max, windowsize, hop = config['dir_max_n'], config['windowsize'], config['hop']
    concat, ext_n, ext_d = config['concat'], ext_conf['num'], ext_conf['direction']
    overlap, too_long, too_short = ext_conf['overlap'], ali_conf['too_long'], ali_conf['too_short']

    def _aligin(input, output):
        _padding = hop if padding else 0
        length = hop * ext_n * ext_d + (windowsize - hop)

        filename, wav, next_wav = None, [], []
        curr_dir_n, curr_samp_n, curr_dir = -1, file_max + 1, os.path.join(output, '0')
        for root, dirs, files in os.walk(input):
            for f in files:
                if f.endswith('.wav'):
                    if filename is None:
                        filename = f

                    y, sr = librosa.load(os.path.join(root, f), sr=samplerate)
                    wav.extend(y)
                    if length > len(wav) + _padding * 2:
                        if concat:
                            continue
                        if too_short == 'append_0':
                            n = length - (len(wav) + _padding * 2)
                            wav += [0 for _ in range(n)]
                        elif too_short == 'discard':
                            filename, wav = None, []
                            continue
                        else:
                            raise ValueError('%s is not supported' % too_short)
                    elif length < len(wav) + _padding * 2:
                        if overlap:
                            surplus = (len(wav) + _padding * 2 - windowsize) % hop
                        else:
                            surplus = (len(wav) + _padding * 2) % length
                        if surplus != 0:
                            if concat:
                                next_wav = wav[-surplus:]
                                del wav[-surplus:]
                            elif too_long == 'append_0':
                                comp = hop - surplus if overlap else length - surplus
                                wav += [0 for _ in range(comp)]
                            elif too_long == 'cut':
                                del wav[-surplus:]
                            else:
                                raise ValueError('%s is not supported' % too_long)

                    if curr_samp_n > file_max:
                        curr_dir_n, curr_samp_n = curr_dir_n + 1, 0
                        curr_dir = os.path.join(output, str(curr_dir_n))
                        util.mkdir_p(curr_dir)
                    path = os.path.join(curr_dir, filename[:-4] + ".voice.wav")
                    librosa.output.write_wav(path, np.array(wav), samplerate)

                    curr_samp_n += 1
                    if concat:
                        filename, wav = None, next_wav
                        next_wav = []
                    else:
                        filename, wav = None, []

    if 'train' in config:
        _aligin(os.path.join(config['voice_root'], 'train'), os.path.join(output_root, 'train', 'raw'))

    if 'test' in config:
        _aligin(os.path.join(config['voice_root'], 'test'), os.path.join(output_root, 'test', 'raw'))

    if 'eval' in config:
        _aligin(os.path.join(config['voice_root'], 'eval'), os.path.join(output_root, 'eval', 'raw'))

    open(os.path.join(output_root, check_aligin_file), 'w').close()
    return output_root


def mix(config):
    existed, code = check_trans_exists(config, check_mix_file)
    if existed:
        return
    existed, code = check_trans_exists(config, check_aligin_file)
    if not existed:
        aligin(config)
    output_root = os.path.join(data_root, code)

    samplerate = config['samplerate']

    def _mix(noise_root, output, noise_part, snr):
        voice_fs = os.listdir(output)
        voice_fs = [f for f in voice_fs if f.endswith('.voice.wav')]
        voice_fs.sort()

        noise_fs = os.listdir(noise_root)
        noise_fs = [f for f in noise_fs if f.endswith('.wav')]
        noise_fs.sort()
        noise_indx = 0

        for voice_f in voice_fs:
            noise_f = noise_fs[noise_indx]
            voice, sr = librosa.load(os.path.join(output, voice_f), sr=samplerate)
            noise, sr = librosa.load(os.path.join(noise_root, noise_f), sr=samplerate)
            if len(noise) < len(voice) * 3:
                noise = np.tile(noise, int(np.ceil(len(voice) * 3 / len(noise))))
            one_third = int(len(noise) / 3)
            noise = noise[one_third * noise_part:one_third * noise_part + len(voice)]

            n_2 = np.sum(noise ** 2)
            if n_2 > 0.:
                a = np.sqrt(np.sum(voice ** 2) / (n_2 * 10 ** (snr / 10)))
                noise = a * noise
            mixture = voice + noise

            mix_name = '%s+%s' % (voice_f[:-10], noise_f[:-4])
            librosa.output.write_wav(os.path.join(output, mix_name + ".noise.wav"), noise, sr=samplerate)
            librosa.output.write_wav(os.path.join(output, mix_name + ".mix.wav"), mixture, sr=samplerate)

            noise_indx = (noise_indx + 1) % len(noise_fs)

    if 'train' in config:
        noise_root = os.path.join(config['noise_root'], 'train')
        path = os.path.join(output_root, 'train', 'raw')
        dirs = os.listdir(path)
        for d in dirs:
            p = os.path.join(path, d)
            if os.path.isdir(p):
                _mix(noise_root, p, 0, config['train']['snr'])

    if 'eval' in config:
        noise_root = os.path.join(config['noise_root'], 'eval')
        path = os.path.join(output_root, 'eval', 'raw')
        dirs = os.listdir(path)
        for d in dirs:
            p = os.path.join(path, d)
            if os.path.isdir(p):
                _mix(noise_root, p, 1, config['eval']['snr'])

    if 'test' in config:
        noise_root = os.path.join(config['noise_root'], 'test')
        raw_p = os.path.join(output_root, 'test', 'raw')
        for snr in config['test']['snr']:
            snr_p = os.path.join(output_root, 'test', str(snr), 'raw')
            if os.path.exists(snr_p):
                shutil.rmtree(snr_p)
            shutil.copytree(raw_p, snr_p)

            dirs = os.listdir(snr_p)
            for d in dirs:
                p = os.path.join(snr_p, d)
                if os.path.isdir(p):
                    _mix(noise_root, p, 2, snr)
    open(os.path.join(output_root, check_mix_file), 'w').close()


def extend(sig, num, direction, overlap):
    # 拓展帧
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
    return extension


def calc_feat(config):
    existed, code = check_trans_exists(config, check_feat_file)
    if existed:
        return
    existed, code = check_trans_exists(config, check_mix_file)
    if not existed:
        mix(config)
    output_root = os.path.join(data_root, code)

    samplerate = config['samplerate']
    ext_conf = config['ext']
    ext_d, overlap = ext_conf['direction'], ext_conf['overlap']
    padding, hop = config['padding'], config['hop']

    def _calc_feat(root, suffix, ext_n, feat):
        i = 0
        while True:
            raw_p = os.path.join(root, 'raw', str(i))
            if not os.path.exists(raw_p):
                break

            feat_p = os.path.join(root, 'feat', str(i))
            util.mkdir_p(feat_p)

            files = os.listdir(raw_p)
            for fname in files:
                if fname.endswith(suffix):
                    wav, sr = librosa.load(os.path.join(raw_p, fname), sr=samplerate)
                    if padding:
                        wav = np.pad(wav, (hop, hop), 'constant', constant_values=(0, 0))
                    trans = compute_feat(wav, feat, config)
                    trans = extend(trans, ext_n, ext_d, overlap)
                    np.save(os.path.join(feat_p, fname[:-4]), trans)
            i += 1

    if 'train' in config:
        root = os.path.join(output_root, 'train')
        _calc_feat(root, '.voice.wav', config['ext']['num.clean'], config['feat']['feat.clean'])
        _calc_feat(root, '.noise.wav', config['ext']['num.noise'], config['feat']['feat.noise'])
        _calc_feat(root, '.mix.wav', config['ext']['num.mix'], config['feat']['feat.mix'])

    if 'eval' in config:
        root = os.path.join(output_root, 'eval')
        _calc_feat(root, '.voice.wav', config['ext']['num.clean'], config['feat']['feat.clean'])
        _calc_feat(root, '.noise.wav', config['ext']['num.noise'], config['feat']['feat.noise'])
        _calc_feat(root, '.mix.wav', config['ext']['num.mix'], config['feat']['feat.mix'])

    if 'test' in config:
        root = os.path.join(output_root, 'test')
        for snr in config['test']['snr']:
            _root = os.path.join(root, str(snr))
            _calc_feat(_root, '.voice.wav', config['ext']['num.clean'], config['feat']['feat.clean'])
            _calc_feat(_root, '.noise.wav', config['ext']['num.noise'], config['feat']['feat.noise'])
            _calc_feat(_root, '.mix.wav', config['ext']['num.mix'], config['feat']['feat.mix'])

    open(os.path.join(output_root, check_feat_file), 'w').close()


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def calc_tfrecord(config):
    # 把数据转换成tfrecord格式的文件，暂时没用这个函数
    existed, code = check_trans_exists(config, check_tfrecord_file)
    if existed:
        return
    existed, code = check_trans_exists(config, check_feat_file)
    if not existed:
        calc_feat(config)
    output_root = os.path.join(data_root, code)

    def _convert(root):
        i = 0
        while True:
            feat_p = os.path.join(root, 'feat', str(i))
            if not os.path.exists(feat_p):
                break

            writer = tf.python_io.TFRecordWriter(feat_p + '.tfrecord')
            files = os.listdir(feat_p)
            files.sort()
            for j in range(0, len(files), 3):
                mix = np.load(os.path.join(feat_p, files[j]))
                noise = np.load(os.path.join(feat_p, files[j + 1]))
                voice = np.load(os.path.join(feat_p, files[j + 2]))
                path = os.path.join(feat_p, files[j])[:-8]
                for m, n, v, idx in zip(mix, noise, voice, range(len(mix))):
                    _path = path + '.' + str(idx)
                    example = tf.train.Example(features=tf.train.Features(feature={
                        'mix': _bytes_feature(m.tostring()),
                        'noise': _bytes_feature(n.tostring()),
                        'voice': _bytes_feature(v.tostring()),
                        'path': _bytes_feature(_path.encode('utf-8'))}))
                    writer.write(example.SerializeToString())
            writer.close()
            i += 1

    if 'train' in config:
        _convert(os.path.join(output_root, 'train'))
    if 'eval' in config:
        _convert(os.path.join(output_root, 'eval'))
    if 'test' in config:
        for snr in config['test']['snr']:
            _convert(os.path.join(output_root, 'test', str(snr)))
    open(os.path.join(output_root, check_tfrecord_file), 'w').close()


def clean_dataset(config, keep_tfrecord=True):
    existed, code = check_trans_exists(config, check_tfrecord_file)
    clean_root = os.path.join(data_root, code)
    if not keep_tfrecord:
        util.rm_r(clean_root)
        return

    def _clean(root):
        util.rm_r(os.path.join(root, 'raw'))
        feat_p = os.path.join(root, 'feat')
        if os.path.isdir(feat_p):
            dirs = os.listdir(feat_p)
            for dir in dirs:
                path = os.path.join(feat_p, dir)
                if os.path.isdir(path):
                    util.rm_r(path)

    util.rm_r(os.path.join(clean_root, check_aligin_file))
    util.rm_r(os.path.join(clean_root, check_mix_file))
    util.rm_r(os.path.join(clean_root, check_feat_file))

    _clean(os.path.join(clean_root, 'train'))
    _clean(os.path.join(clean_root, 'eval'))
    util.rm_r(os.path.join(clean_root, 'test', 'raw'))
    for snr in config['test']['snr']:
        _clean(os.path.join(clean_root, 'test', str(snr)))

        # aligin(config)
