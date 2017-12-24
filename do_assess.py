# -*- coding: utf-8 -*-
"""
@author: PengChuan
"""
import os
import librosa
from assess.core import calc_stoi, calc_pesq, calc_ssnr


def eval_dir(enh_root, cln_root, msr_list, sr=16000, align=False):
    '''
    msr是列表 e.g., ['stoi']
    评价指标的计算
    '''
    # 该函数这里没有用到
    files = os.listdir(enh_root)

    score = [0. for _ in range(len(msr_list))]
    count = 0
    for file in files:
        if not file.endswith('-sep.wav'):
            continue

        enh, _ = librosa.load(os.path.join(enh_root, file), sr=sr)
        cln_name = file[:file.index('+')] + '.wav'
        cln, _ = librosa.load(os.path.join(cln_root, cln_name), sr=sr)

        if align:
            if len(enh) < len(cln):
                cln = cln[:len(enh)]
            elif len(enh) > len(cln):
                enh = enh[:len(cln)]

        for i in range(len(msr_list)):
            if msr_list[i] == 'stoi':
                score[i] += calc_stoi(cln, enh, sr)
            elif msr_list[i] == 'pesq':
                score[i] +=  calc_pesq(cln, enh, sr, is_file=False)
            elif msr_list[i] == 'ssnr':
                score[i] += calc_ssnr(cln, enh, int(sr*0.02))
        count += 1

    return [s / count for s in score]

if __name__ == '__main__':
    import sys
    enh_root = sys.argv[1]
    cln_root = sys.argv[2]
    msr_list = sys.argv[3:]

    score = eval_dir(enh_root, cln_root, msr_list, align=True)
    print(score)