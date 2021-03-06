import logging
import os

import numpy as np
import tensorflow as tf

import data.gen.dnn
import nnet.conf.dnn as config
import util
from nnet import nn_model, evaluate


class BasicNN(object):
    def __init__(self):
        in_dim = config.INPUT_DIM # 输入的维度
        out_dim = config.OUTPUT_DIM # 输出的维度
        ext_num = config.EXTEND_NUM # 神经网络输入，每帧前/后扩展数
        logging.info('input/output dim: %d/%d' % (in_dim, out_dim))

        graph = tf.Graph()  # tensorflow使用图 (graph) 来表示计算任务
        with graph.as_default():
            # Build model structure
            mix_feat = tf.placeholder("float", [None, ext_num * 2 + 1, in_dim])
            clean = tf.placeholder("float", [None, out_dim])
            mix_mag = tf.placeholder("float", [None, out_dim]) #mix_magnitude
            keep = nn_model.get_dropout_placeholder(config.NN_STRUCT) # 占位符

            # 根据网络结构堆叠各网络层
            layer = nn_model.build(config.NN_STRUCT, mix_feat, keep, True)

            # enhance 增强
            enhed = self.tf_mask(layer, mix_mag)

            self.graph = graph
            self.mix_feat, self.clean, self.mix_mag = mix_feat, clean, mix_mag
            self.enhed = enhed
            self.cost, self.optimizer = None, None
            self.ext_num = ext_num
            self.keep = keep
            self.best_score = None

    def tf_mask(self, g_out, mix_mag):
        # 二值掩蔽 黄培森
        out_dim = g_out.get_shape().as_list()[1] // 2 # //表示整数除法
        est_y1 = tf.abs(g_out[:, :out_dim])
        est_y2 = tf.abs(g_out[:, out_dim:])
        return est_y1 / (est_y1 + est_y2 + 1e-8) * mix_mag

    def gen_optimizer(self):
        # 生成优化器
        # 训练的时候用来做梯度下降，训练模型
        logging.info('learning rate: %g' % config.LEARNING_RATE)
        logging.info('weight L2 regularization: %g' % config.L2_REGULAR)
        # Define loss and optimizer
        with self.graph.as_default():
            cost = tf.reduce_mean(tf.square(self.enhed - self.clean))
            # cost = tf.reduce_mean(tf.abs(self.out_y1 - self.y1))

            if config.L2_REGULAR > 0.:
                theta = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                wl2 = tf.add_n([tf.nn.l2_loss(v) for v in theta])
                cost += wl2 * config.L2_REGULAR

            optimizer = tf.train.AdamOptimizer(learning_rate=config.LEARNING_RATE)
            if not config.CLIP_WEIGHT:
                optimizer = optimizer.minimize(cost)
            else:
                gvs = optimizer.compute_gradients(cost)
                # 对梯度做上下限，应对梯度爆炸
                capped_gvs = [(tf.clip_by_value(grad, -1, 1), var) for grad, var in gvs]
                optimizer = optimizer.apply_gradients(capped_gvs)

            self.optimizer = optimizer
            self.cost = cost

    def train(self, restore=True):
        logging.info("train SENN with max iteration: %g" % config.MAX_ITERATION) # log模型训练次数
        logging.info('model save path: %s' % config.MODEL_PATH) # 模型保存路径

        tr_gen_g = data.gen.dnn.Generator(config.data, 'train') #
        ev_gen_g = data.gen.dnn.Generator(config.data, 'eval') # 求值

        with self.graph.as_default():
            saver = tf.train.Saver() # Saves and restores variables.
            # Launch the graph
            with tf.Session() as sess:
                eval_runner = InnerEvalRunner(self, sess)
                # Initializing the variables
                if restore:
                    saver.restore(sess, config.MODEL_PATH)
                else:
                    sess.run(tf.global_variables_initializer())
                self.best_score = - np.inf
                for epoch in range(1, config.MAX_ITERATION + 1):
                    cost_sum, count = 0., 0.
                    for bat in tr_gen_g.asyn_gen():
                        feed = {self.mix_feat: bat.mix_feat, self.clean: bat.clean, self.mix_mag: bat.mix}
                        nn_model.feed_dropout_keep_prob(feed, self.keep)
                        _, c = sess.run(fetches=[self.optimizer, self.cost], feed_dict=feed)
                        count += 1
                        cost_sum += c
                    if epoch % 1 == 0 or epoch <= 1:
                        evaler = evaluate.Evaler(ev_gen_g, eval_runner)
                        mean_stoi, mean_pesq, mean_ssnr = evaler.run_eval()
                        logging.info("Epoch: %04d cost=%.9f stoi=%.4f pesq=%.4f ssnr=%.4f" %
                                     (epoch, cost_sum, mean_stoi, mean_pesq, mean_ssnr))

                        if mean_stoi > self.best_score:
                            saver.save(sess, config.MODEL_PATH)
                            self.best_score = mean_stoi
                logging.info("Optimization Done! best eval score: %.9f" % self.best_score)

    def test(self):
        with self.graph.as_default():
            saver = tf.train.Saver()
            with tf.Session() as sess:
                saver.restore(sess, config.MODEL_PATH)
                eval_runner = InnerEvalRunner(self, sess)
                for snr in config.data['test']['snr']:
                    dataset = data.gen.dnn.Generator(config.data, 'test', snr)
                    root = os.path.join(dataset.vpath, 'test.out', str(snr))
                    post_runner = TestPostRunner(root, config.data['samplerate'])
                    tester = evaluate.Evaler(dataset, eval_runner, post_runner)
                    mean_stoi, mean_pesq, mean_ssnr = tester.run_eval()
                    logging.info('snr=%3d: stoi=%.4f, pesq=%.4f, ssnr=%.4f' %
                                 (snr, mean_stoi, mean_pesq, mean_ssnr))
                logging.info('Test output saved in %s' % os.path.dirname(root))


class InnerEvalRunner(object):
    def __init__(self, model, session):
        self.sess = session
        self.model = model

    def run(self, sample):
        feed = {self.model.mix_feat: sample.mix_feat, self.model.mix_mag: sample.mix}
        nn_model.feed_dropout_keep_prob(feed, self.model.keep, disable=True)
        return self.sess.run(self.model.enhed, feed_dict=feed)


class TestPostRunner(object):
    def __init__(self, root, samplerate):
        self.root = root
        self.sr = samplerate
        util.mkdir_p(root)

    def post_run(self, sample):
        v_name = os.path.basename(sample.vpath)
        n_name = os.path.basename(sample.npath)
        mix_name = "%s+%s.wav" % (v_name[:-4], n_name[:-4])
        enh_name = "%s+%s.enh.wav" % (v_name[:-4], n_name[:-4])
        util.wav_write(os.path.join(self.root, mix_name), sample.mix_raw, self.sr, norm=True)
        util.wav_write(os.path.join(self.root, enh_name), sample.enhed_raw, self.sr, norm=True)
