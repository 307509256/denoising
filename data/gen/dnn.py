# -*- coding: utf-8 -*-
"""
@author: PengChuan
"""
from data.gen_data import Generator as BasicGenerator
from data.gen_data import adj_extend


class Generator(BasicGenerator):
    # 从数据集读入数据，预处理，为train，eval，test做准备
    # 百度训练、验证、测试！！！！！！！！！！！
    def __init__(self, config, phase, snr=None):
        super().__init__(config, phase, snr)
        self.n_ext = self.conf['ext']['num'] # 配置延拓的采样点

    def gen_with_path(self, vpath, npath):
        sp = super().gen_with_path(vpath, npath)
        ext_conf = self.conf['ext']
        ext_d, overlap = ext_conf['direction'], ext_conf['overlap']
        sp.mix_feat = adj_extend(sp.mix_feat, self.n_ext, ext_d, overlap)
        return sp
