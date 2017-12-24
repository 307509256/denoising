# -*- coding: utf-8 -*-
"""
@author: PengChuan
"""
from data.gen_data import Generator as BasicGenerator
from data.gen_data import adj_extend


class Generator(BasicGenerator):
    def __init__(self, config, phase, g_ext, d_ext, snr=None):
        super().__init__(config, phase, snr)
        self.g_ext, self.d_ext = g_ext, d_ext

    def gen_with_path(self, vpath, npath):
        sp = super().gen_with_path(vpath, npath)
        ext_conf = self.conf['ext']
        ext_d, overlap = ext_conf['direction'], ext_conf['overlap']

        sp.mix_feat = adj_extend(sp.mix_feat, self.g_ext, ext_d, overlap)
        sp.mix = adj_extend(sp.mix, self.d_ext, ext_d, overlap)
        sp.clean = adj_extend(sp.clean, self.d_ext, ext_d, overlap)

        return sp
