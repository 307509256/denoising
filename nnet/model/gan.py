# -*- coding: utf-8 -*-
"""
@author: PengChuan
"""

import logging
import os

import numpy as np
import tensorflow as tf

import nnet.conf.gan as config
from data.gen import gan as gen_data
from nnet import nn_model, evaluate
from nnet.model.dnn import InnerEvalRunner as BasicInnerEvalRunner
from nnet.model.dnn import TestPostRunner


class GAN:
    def __init__(self):
        in_dim = config.INPUT_DIM
        out_dim = config.OUTPUT_DIM
        g_extend = config.G_EXTEND
        d_extend = config.D_EXTEND
        logging.info('G input/output dim: %d/%d' % (in_dim, out_dim))
        logging.info('G/D frame num: %d/%d' % (g_extend * 2 + 1, d_extend * 2 + 1))
        logging.info('condition gan: %s' % config.CONDITIONAL_GAN)

        graph = tf.Graph()
        with graph.as_default():
            # mix_feat, ext_mix_mag, clean, phase = self._read_data(g_extend, d_extend, in_dim, out_dim)
            mix_feat = tf.placeholder("float", [None, g_extend * 2 + 1, in_dim]) #混合语音的feat
            ext_mix_mag = tf.placeholder("float", [None, d_extend * 2 + 1, out_dim]) # 扩展的mix幅度谱
            clean = tf.placeholder("float", [None, d_extend * 2 + 1, config.MAG_DIM])

            g_keep = nn_model.get_dropout_placeholder(config.G_NN_STRUCT)
            d_keep = nn_model.get_dropout_placeholder(config.D_NN_STRUCT)

            # G net
            logging.info('build GAN G =======================')
            with tf.variable_scope('generator'):
                enhed = nn_model.build(config.G_NN_STRUCT, mix_feat, g_keep, log_struct=True)
                out_y1 = enhed

            ext_out_y1 = tf.expand_dims(out_y1, axis=1)
            if d_extend > 0:
                ext_out_y1 = tf.concat((clean[:, :d_extend], ext_out_y1, clean[:, -d_extend:]), 1)
            ext_clean = clean

            # D net
            logging.info('build GAN D ===============================')
            with tf.variable_scope('critic'):
                if config.CONDITIONAL_GAN:
                    ext_clean = tf.concat([ext_clean, ext_mix_mag], axis=2)
                    ext_clean = tf.reshape(ext_clean, [-1, d_extend * 2 + 1, out_dim, 2])
                true_logit = nn_model.build(config.D_NN_STRUCT, ext_clean, d_keep, log_struct=True)
            with tf.variable_scope('critic', reuse=True):
                if config.CONDITIONAL_GAN:
                    ext_out_y1 = tf.concat([ext_out_y1, ext_mix_mag], axis=2)
                    ext_out_y1 = tf.reshape(ext_out_y1, [-1, d_extend * 2 + 1, out_dim, 2])
                fake_logit = nn_model.build(config.D_NN_STRUCT, ext_out_y1, d_keep)
            true_logit = tf.clip_by_value(true_logit, -1e6, 1e6)
            fake_logit = tf.clip_by_value(fake_logit, -1e6, 1e6)

        self.graph = graph
        self.mix_feat, self.mix_mag = mix_feat, ext_mix_mag
        self.g_out, self.in_clean = out_y1, clean
        self.true_samp, self.fake_samp = ext_clean, ext_out_y1
        self.true_logit, self.fake_logit = true_logit, fake_logit
        self.c_loss, self.g_loss = None, None
        self.opt_c, self.opt_g = None, None
        self.best_score = None
        self.d_samp_i, self.samp_num, self.samp_ii = None, -1, -1
        self.emd = None
        self.grad_penalty = None
        self.g_ext, self.d_ext = g_extend, d_extend
        self.g_keep, self.d_keep = g_keep, d_keep
        self.enhed = enhed
        self.d_samp_empty = None

    def gen_optimizer(self):
        logging.info('G, D learning rate: %g, %g' % (config.G_LR, config.D_LR))
        logging.info("gradient penalty: %s, lambda: %g" %
                     (config.GRADIENT_PENALTY, config.GP_LAMBDA))
        logging.info("Adam optimizer: %s" % config.OPT_ADAM)
        logging.info('G, D L2 regularization: %g, %g' %
                     (config.G_L2_REGULAR, config.D_L2_REGULAR))

        # Define loss and optimizer
        with self.graph.as_default():
            emd = tf.reduce_mean(tf.square(self.fake_logit - self.true_logit))

            fake_logit, true_logit = self.fake_logit, self.true_logit
            fake_samp, true_samp = self.fake_samp, self.true_samp
            d_loss = tf.reduce_mean(fake_logit - true_logit)

            grad_penalty = None
            if config.GRADIENT_PENALTY:
                alpha_dist = tf.contrib.distributions.Uniform(0., 1.)
                _shape = tf.shape(fake_samp)
                if config.CONDITIONAL_GAN:
                    alpha = alpha_dist.sample([_shape[0], 1, 1, 1])
                else:
                    alpha = alpha_dist.sample([_shape[0], 1, 1])
                interpolated = true_samp + alpha * (fake_samp - true_samp)
                mid = interpolated[:, self.d_ext, :]
                with tf.variable_scope('critic', reuse=True):
                    inte_logit = nn_model.build(config.D_NN_STRUCT, interpolated, mid, self.d_keep)
                gradients = tf.gradients(inte_logit, [interpolated, ])[0]
                grad_l2 = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=1))
                grad_penalty = tf.reduce_mean(tf.square(grad_l2 - 1))
            g_loss = tf.reduce_mean(-self.fake_logit)

            if config.LS_GAN:
                d_loss = tf.reduce_mean(tf.square(fake_logit + 1) + tf.square(true_logit - 1))
                g_loss = tf.reduce_mean(tf.square(self.fake_logit))

            theta_c = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic')
            theta_g = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')

            if config.G_L2_REGULAR > 0.:
                wl2 = tf.add_n([tf.nn.l2_loss(v) for v in theta_g])
                g_loss += wl2 * config.G_L2_REGULAR
            if config.D_L2_REGULAR > 0.:
                wl2 = tf.add_n([tf.nn.l2_loss(v) for v in theta_c])
                d_loss += wl2 * config.D_L2_REGULAR
            if config.G_L1_REGULAR > 0:
                g_loss += tf.reduce_mean(tf.abs((self.fake_samp - self.true_samp))) * config.G_L1_REGULAR
            if config.OPT_ADAM:
                opt_d = tf.train.AdamOptimizer(learning_rate=config.D_LR)
                opt_g = tf.train.AdamOptimizer(learning_rate=config.G_LR)
            else:
                opt_d = tf.train.RMSPropOptimizer(learning_rate=config.D_LR)
                opt_g = tf.train.RMSPropOptimizer(learning_rate=config.G_LR)

            if config.GRADIENT_PENALTY:
                opt_d = opt_d.minimize(d_loss + config.GP_LAMBDA * grad_penalty, var_list=theta_c)
            else:
                opt_d = opt_d.minimize(d_loss, var_list=theta_c)
            opt_g = opt_g.minimize(g_loss, var_list=theta_g)

        self.grad_penalty = grad_penalty
        self.g_loss = g_loss
        self.opt_g = opt_g
        self.emd = emd
        self.d_samp_empty = tf.size(fake_logit) <= 0
        self.d_loss = d_loss
        self.opt_d = opt_d
        self.best_score = None

    def get_feed(self, dataset):
        feed = {}
        dataset.asyn_gen()

    def train(self, restore=True):
        logging.info('G max iteration: %d, D iter %d times per G iter' %
                     (config.MAX_ITERATION, config.D_ITER))
        logging.info('model save path: %s' % config.MODEL_PATH)

        tr_gen_g = gen_data.Generator(config.data, 'train', self.g_ext, self.d_ext)
        ev_gen_g = gen_data.Generator(config.data, 'eval', self.g_ext, self.d_ext)

        with self.graph.as_default():
            saver = tf.train.Saver()
            with tf.Session() as sess:
                eval_runner = InnerEvalRunner(self, sess)
                if restore:
                    logging.info('restore model from %s' % restore)
                    saver.restore(sess, config.MODEL_PATH)
                else:
                    sess.run(tf.global_variables_initializer())
                self.best_score, = - np.inf,
                for epoch in range(1, config.MAX_ITERATION + 1):
                    cost_d, cost_g, emd = 0., 0., 0.
                    count = 0.
                    for bat in tr_gen_g.asyn_gen():
                        count += 1
                        if count % config.D_ITER > 0:
                            feed = {self.mix_feat: bat.mix_feat, self.mix_mag: bat.mix, self.in_clean: bat.clean}
                            nn_model.feed_dropout_keep_prob(feed, self.g_keep, disable=True)
                            nn_model.feed_dropout_keep_prob(feed, self.d_keep)
                            _, c = sess.run(fetches=[self.opt_d, self.d_loss], feed_dict=feed)
                            cost_d += c
                        else:
                            feed = {self.mix_feat: bat.mix_feat, self.mix_mag: bat.mix, self.in_clean: bat.clean}
                            nn_model.feed_dropout_keep_prob(feed, self.g_keep)
                            nn_model.feed_dropout_keep_prob(feed, self.d_keep, disable=True)
                            _, c, _emd = sess.run(fetches=[self.opt_g, self.g_loss, self.emd], feed_dict=feed)
                            cost_g += c
                            emd += _emd

                    cost_d /= max(1, count) / (config.D_ITER + 1) * config.D_ITER
                    cost_g /= max(1., count) / (config.D_ITER + 1)
                    emd /= max(1., count) / (config.D_ITER + 1)
                    if epoch % 5 == 0 or epoch <= 1:
                        evaler = evaluate.Evaler(ev_gen_g, eval_runner)
                        mean_stoi, mean_pesq, mean_ssnr = evaler.run_eval()
                        logging.info("Epoch: %04d d_cost=%.4f g_cost=%.4f emd=%.4f stoi=%.4f pesq=%.4f "
                                     "ssnr=%.4f" % (epoch, cost_d, cost_g, emd, mean_stoi, mean_pesq, mean_ssnr))

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
                    dataset = gen_data.Generator(config.data, 'test', self.g_ext, self.d_ext, snr)
                    root = os.path.join(dataset.vpath, 'test.out', str(snr))
                    post_runner = TestPostRunner(root, config.data['samplerate'])
                    tester = evaluate.Evaler(dataset, eval_runner, post_runner)
                    mean_stoi, mean_pesq, mean_ssnr = tester.run_eval()
                    logging.info('snr=%3d: stoi=%.4f, pesq=%.4f, ssnr=%.4f' %
                                 (snr, mean_stoi, mean_pesq, mean_ssnr))
                logging.info('Test output saved in %s' % os.path.dirname(root))


class InnerEvalRunner(BasicInnerEvalRunner):
    def __init__(self, gan, session):
        super().__init__(gan, session)

    def run(self, sample):
        feed = {self.model.mix_feat: sample.mix_feat}
        nn_model.feed_dropout_keep_prob(feed, self.model.g_keep, disable=True)
        nn_model.feed_dropout_keep_prob(feed, self.model.d_keep, disable=True)
        return self.sess.run(self.model.enhed, feed_dict=feed)
