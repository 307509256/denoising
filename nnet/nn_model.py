# -*- coding: utf-8 -*-
"""
@author: PengChuan
"""

import json
import logging
import tensorflow as tf
import tensorflow.contrib.layers as ly
from tensorflow.contrib import rnn


def ly_dropout(layer, keep_prob):
    return tf.nn.dropout(layer, keep_prob)


def ly_fully_connected(layer, struct):
    # 每一层，配置，
    norm_fn = struct['norm_fn'] if 'norm_fn' in struct else None
    w_init = struct['init'] if 'init' in struct else ly.xavier_initializer()
    fun = struct['fun'] if 'fun' in struct else None
    return ly.fully_connected(layer, struct['n_out'], activation_fn=fun,
                              normalizer_fn=norm_fn, weights_initializer=w_init)


def ly_conv(layer, struct):
    # 卷积
    padding = struct['pad'] if 'pad' in struct else 'SAME'
    norm_fn = struct['norm_fn'] if 'norm_fn' in struct else None
    w_init = struct['init'] if 'init' in struct else ly.xavier_initializer()
    fun = struct['fun'] if 'fun' in struct else None
    dform = struct['dform'] if 'dform' in struct else 'NHWC'
    stride = struct['stride'] if 'stride' in struct else 1
    return ly.conv2d(layer, num_outputs=struct['n_out'], kernel_size=struct['ksize'],
                     stride=stride, activation_fn=fun, padding=padding,
                     normalizer_fn=norm_fn, weights_initializer=w_init, data_format=dform)


def ly_pool(layer, struct):
    # 池化
    padding = struct['pad'] if 'pad' in struct else 'SAME'
    dform = struct['dform'] if 'dform' in struct else 'NHWC'
    # return tf.nn.max_pool(layer, ksize=struct['ksize'], strides=struct['stride'],
    #                       padding=padding)
    return tf.nn.pool(layer, struct['ksize'], 'MAX', padding, strides=struct['stride'],
                      data_format=dform)


def ly_rnn(layer, struct, initial_state=None):
    for _ in range(len(layer.get_shape()), 3):
        layer = tf.expand_dims(layer, axis=0)

    cell = rnn.BasicLSTMCell(struct['n_out'], activation=struct['fun'])
    # cell = rnn.BasicRNNCell(layers_struct['n_out'], activation=layers_struct['fun'])

    output, state = tf.nn.dynamic_rnn(cell, layer, dtype=tf.float32, initial_state=initial_state)

    return output, state


# def stack_rnn(input_layer, layers_struct, curr_indx):
#     # {'type': rnn.T_NN_RECURRENT, 'num': 1024},  # 循环连接层
#
#     prev = layers_struct[curr_indx - 1]
#     input_layer = tf.reshape(input_layer, [1, -1, prev['num']])  # 扩展成三阶张量
#
#     stack_layers = []
#     while True:
#         curr = layers_struct[curr_indx]
#         with tf.variable_scope('%s%d' % ('rnn', curr_indx)):
#             if curr_indx >= len(layers_struct) - 1:  # 最后一层没有非线性激活函数
#                 act = tf.identity
#             else:
#                 act = tf.nn.relu
#             cell = rnn.BasicLSTMCell(curr['num'], activation=act)
#             # cell = rnn.BasicRNNCell(curr['num'], activation=act)
#         stack_layers.append(cell)
#
#         if curr_indx <= len(layers_struct) - 2 \
#                 and layers_struct[curr_indx + 1]['type'] == 'rnn':
#             curr_indx += 1
#         else:
#             break
#
#     if len(stack_layers) > 1:
#         cell = rnn.MultiRNNCell(stack_layers)
#
#     output, state = tf.nn.dynamic_rnn(cell, input_layer, dtype=tf.float32)
#     output = tf.reshape(output, [-1, curr['num']])  # 变换成二阶张量
#
#     return output, state, curr_indx


def get_dropout_placeholder(nn_struct):
    keep_probs = []
    for stct in nn_struct:
        if stct['type'] == 'dropout':
            pl_kp = tf.placeholder("float")
            kp = stct['keep']
            keep_probs.append((pl_kp, kp,))
    return keep_probs


def feed_dropout_keep_prob(feed, keep_prob, disable=False):
    if not keep_prob:
        return

    for k, v in keep_prob:
        feed[k] = 1. if disable else v


def ly_conv_transpose(layer, struct):
    padding = struct['pad'] if 'pad' in struct else 'SAME'
    norm_fn = struct['norm_fn'] if 'norm_fn' in struct else None
    w_init = struct['init'] if 'init' in struct else ly.xavier_initializer()
    fun = struct['fun'] if 'fun' in struct else None
    conv_ly = ly.conv2d_transpose(layer, num_outputs=struct['n_out'], kernel_size=struct['ksize'],
                     stride=struct['stride'], activation_fn=fun, padding=padding,
                     normalizer_fn=norm_fn, weights_initializer=w_init)
    if 'out_shape' in struct:
        out_shape = struct['out_shape']
        begin, size = [], []
        for o, c in zip(out_shape, conv_ly.get_shape().as_list()):
            if o <= 0:
                begin.append(0)
                size.append(-1)
            elif o <= c:
                begin.append((c-o)//2)
                size.append(o)
            else:
                raise ValueError("conv2d transpose cannot cut output shape %s from %s" %
                                 (out_shape, conv_ly.get_shape().as_list()))
        conv_ly = tf.slice(conv_ly, begin, size)

    return conv_ly


def batch_to_time(value, dilation):
    # https://github.com/ibab/tensorflow-wavenet/blob/master/wavenet/ops.py
    # 谷歌的语音合成系统 batch是tensor里面的东西
    #
    shape = value.get_shape().as_list()
    prepared = tf.reshape(value, [dilation, -1, shape[2]])
    transposed = tf.transpose(prepared, perm=[1, 0, 2])
    return tf.reshape(transposed, [-1, dilation*shape[1], shape[2]])


def time_to_batch(value, dilation):
    shape = value.get_shape().as_list()
    pad_elements = dilation - 1 - (shape[1] + dilation - 1) % dilation
    padded = tf.pad(value, [[0, 0], [0, pad_elements], [0, 0]])
    reshaped = tf.reshape(padded, [-1, dilation, shape[2]])
    transposed = tf.transpose(reshaped, perm=[1, 0, 2])
    return tf.reshape(transposed, [-1, (shape[1]+pad_elements) // dilation, shape[2]])


def ly_conv1d(value, struct):
    # 一维卷积
    padding = struct['pad'] if 'pad' in struct else 'same'
    w_init = struct['init'] if 'init' in struct else ly.xavier_initializer()
    dilation = struct['dilation'] if 'dilation' in struct else 1
    stride = struct['stride'] if 'stride' in struct else 1
    fun = struct['fun'] if 'fun' in struct else None
    dform = struct['dform'] if 'dform' in struct else 'channels_last'
    if dilation != 1 and stride != 1:
        raise ValueError("Dilation(%d) != 1 and stride(%d) != 1 aren't supported" %
                         (dilation, stride))
    # dilation = 1
    # if dilation > 1:
    #     value = time_to_batch(value, dilation)
    value = tf.layers.conv1d(value, struct['n_out'], struct['ksize'], strides=stride,
                             activation=fun, padding=padding, kernel_initializer=w_init,
                             dilation_rate=dilation, data_format=dform)
    # if dilation > 1:
    #     value = batch_to_time(value, dilation)
    return value


def ly_atrous_conv1d(value, struct):
    dilation = struct['dilation']
    if dilation > 1:
        transformed = time_to_batch(value, dilation)
        conv = ly_conv1d(transformed, struct)
        restored = batch_to_time(conv, dilation)
    else:
        restored = ly_conv1d(value, struct)
    return restored


def create_variable(name, shape):
    '''Create a convolution filter variable with the specified name and shape,
    and initialize it using Xavier initialition.'''
    initializer = tf.contrib.layers.xavier_initializer_conv2d()
    variable = tf.Variable(initializer(shape=shape), name=name)
    return variable


def build(nn_struct, layer, keep_probs, log_struct=False):
    # 根据网络结构堆叠各网络层
    if log_struct:
        logging.info("input shape: %s" % layer.get_shape().as_list())

    kp_i = 0

    for stct in nn_struct:
        if stct['type'] == 'dropout':
            layer = ly_dropout(layer, keep_probs[kp_i][0])
            kp_i += 1
        elif stct['type'] == 'full':
            layer = ly.flatten(layer)
            layer = ly_fully_connected(layer, stct)
        elif stct['type'] == 'conv':
            for _ in range(len(layer.get_shape()), 4):
                layer = tf.expand_dims(layer, axis=-1)
            layer = ly_conv(layer, stct)
        elif stct['type'] == 'convT':
            for _ in range(len(layer.get_shape()), 4):
                layer = tf.expand_dims(layer, axis=-1)
            layer = ly_conv_transpose(layer, stct)
        elif stct['type'] == 'pool':
            layer = ly_pool(layer, stct)
        elif stct['type'] == 'reshape':
            layer = tf.reshape(layer, stct['shape'])
        elif stct['type'] == 'conv1d':
            layer = ly_conv1d(layer, stct)
        elif stct['type'] == 'time2batch':
            layer = time_to_batch(layer, stct['dilation'])
        elif stct['type'] == 'batch2time':
            layer = batch_to_time(layer, stct['dilation'])
        elif stct['type'] == 'atrous_conv1d':
            layer = ly_atrous_conv1d(layer, stct)
        elif stct['type'] == 'rnn':
            layer, state = ly_rnn(layer, stct)
        else:
            raise ValueError('not supported NN type: %s' % stct['type'])

        log_layer(stct, layer, log_struct)
    return layer


def ly_wave(layer, struct, out_dim=-1):
    '''
           |-> [gate]   -|        |-> 1x1 conv -> skip output
           |             |-> (*) -|
    input -|-> [filter] -|        |-> 1x1 conv -|
           |                                    |-> (+) -> dense output
           |------------------------------------|
    {'type': 'wave', 'dilation': 2, 'n_out': 1, 'ksize': 3},
    wavenet layer谷歌语音合成
    '''
    in_dim = layer.get_shape().as_list()[2]
    if out_dim == -1:
        out_dim = in_dim

    conv1 = ly_conv1d(layer, {"n_out": struct['n_out'] * 2, 'ksize':struct['ksize'],
                              'dilation': struct['dilation']})
    conv_filter = conv1[:, :, :struct['n_out']]
    conv_gate = conv1[:, :, struct['n_out']:]
    out = tf.tanh(conv_filter) * tf.sigmoid(conv_gate)

    conv2 = ly_conv1d(out, {"n_out": in_dim+out_dim, 'ksize':1})
    residual = conv2[:, :, :in_dim]
    skip_out = conv2[:, :, in_dim:]

    # residual = ly_conv1d(out, {"n_out": in_dim, 'ksize':1})
    # skip_out = ly_conv1d(out, {"n_out": out_dim, 'ksize':1})
    return residual, skip_out + layer # 这表示什么？


def build_wavenet(nn_struct, layer, log_struct=False):
    '''
        WSE_STRUCT = [
        {'type': 'global', 'skip_out':2},
        {'type': 'conv1d', 'n_out':1, 'ksize': 2, 'pad': 'same'},
        {'type': 'wave', 'dilation': 2, 'n_out': 1, 'ksize': 3},
        {'type': 'conv1d', 'dilation': 2, 'n_out': 1, 'ksize': 3},
        ]
    '''
    ly_input = layer
    if log_struct:
        logging.info("input shape: %s" % layer.get_shape().as_list())
    assert nn_struct[0]['type'] == 'global'
    log_layer(nn_struct[0], layer, log_struct)

    skip_out_dim = nn_struct[0]['skip_out']
    skip_cut = nn_struct[0]['skip_cut']
    # skip_w = create_variable('skip_w', [1])
    # skip_out = [skip_w * ly_input[:, skip_cut[0]:skip_cut[1], :]]
    skip_out = []

    ly_1st = nn_struct[1]
    layer = ly_conv1d(layer, {'n_out': ly_1st['n_out'], 'ksize': ly_1st['ksize']})
    log_layer(ly_1st, layer, log_struct)

    for i in range(2, len(nn_struct)):
        struct = nn_struct[i]

        if struct['type'] == 'wave':
            layer, ly_skip = ly_wave(layer, struct, out_dim=skip_out_dim)
            ly_skip = ly_skip[:, skip_cut[0]:skip_cut[1], :]
            skip_w = create_variable('skip_w', [1])
            # skip_out.append(ly_skip * skip_w)
            if log_struct:
                logging.info('skip shape: %s ' % ly_skip.get_shape().as_list())
            skip_out.append(ly_skip)
        elif struct['type'] == 'conv1d':
            if nn_struct[i-1]['type'] == 'wave':
                layer = tf.nn.relu(sum(skip_out))
            layer = ly_conv1d(layer, struct)
        elif struct['type'] == 'full':
            if nn_struct[i-1] == 'wave':
                layer = tf.nn.relu(sum(skip_out))
            layer = ly.flatten(layer)
            layer = ly_fully_connected(layer, struct)

        log_layer(struct, layer, log_struct)

    # w1 = tf.get_variable('mix_w', shape=[1], initializer=tf.ones_initializer())
    # w2 = tf.get_variable('sum_w', shape=[1], initializer=tf.zeros_initializer())
    layer = ly_input[:, skip_cut[0]:skip_cut[1], :] + layer

    l_out = tf.squeeze(layer, -1)
    l_out = ly_fully_connected(l_out, {'n_out': 1, 'fun': None})
    logging.info('output shape: %s ' % l_out.get_shape().as_list())
    return l_out


def log_layer(struct, layer, log_struct=True):
    if log_struct:
        _stct = struct.copy()
        for key in _stct:
            if callable(_stct[key]):
                _stct[key] = _stct[key].__name__
        logging.info('NN struct -> shape: %s --> %s' %
                     (json.dumps(_stct, sort_keys=True), layer.get_shape().as_list()))

