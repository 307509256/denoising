# -*- coding: utf-8 -*-
"""
@author: PengChuan
"""

# import os
# os.environ["CUDA_VISIBLE_DEVICES"]=""
import global_config
import nnet.model.dnn as dnn_model
import nnet.model.gan as gan_model
import nnet.model.rnn as rnn_model

if __name__ == '__main__':
    global_config.config_log()
    
    global_config.NUM_PROCESS = 1 # 禁用多进程，windows此句不注释
	
    gan = gan_model.GAN()
    gan.gen_optimizer()
    gan.train(restore=False)
    gan.test()

    dnn = dnn_model.BasicNN()
    dnn.gen_optimizer()
    dnn.train(restore=False)
    dnn.test()

    rnn = rnn_model.LSTM()
    rnn.gen_optimizer()
    rnn.train(restore=False)
    rnn.test()
