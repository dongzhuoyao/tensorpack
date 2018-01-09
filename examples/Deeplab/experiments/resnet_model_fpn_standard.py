#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: resnet_model.py

import tensorflow as tf
from tensorflow.contrib.layers import variance_scaling_initializer


from tensorpack.tfutils.argscope import argscope, get_arg_scope
from tensorpack.models import (
    Conv2D, GlobalAvgPooling, BatchNorm, BNReLU, FullyConnected,
    LinearWrap, AtrousConv2D, MaxPooling, Deconv2D)


def resnet_shortcut(l, n_out, stride, nl=tf.identity):
    data_format = get_arg_scope()['Conv2D']['data_format']
    n_in = l.get_shape().as_list()[1 if data_format == 'NCHW' else 3]
    if n_in != n_out:   # change dimension when channel is not the same
        return Conv2D('convshortcut', l, n_out, 1, stride=stride, nl=nl)
    else:
        return l


def apply_preactivation(l, preact):
    if preact == 'bnrelu':
        shortcut = l    # preserve identity mapping
        l = BNReLU('preact', l)
    else:
        shortcut = l
    return l, shortcut


def get_bn(zero_init=False):
    """
    Zero init gamma is good for resnet. See https://arxiv.org/abs/1706.02677.
    """
    if zero_init:
        return lambda x, name: BatchNorm('bn', x, gamma_init=tf.zeros_initializer())
    else:
        return lambda x, name: BatchNorm('bn', x)


def preresnet_basicblock(l, ch_out, stride, preact):
    l, shortcut = apply_preactivation(l, preact)
    l = Conv2D('conv1', l, ch_out, 3, stride=stride, nl=BNReLU)
    l = Conv2D('conv2', l, ch_out, 3)
    return l + resnet_shortcut(shortcut, ch_out, stride)


def preresnet_bottleneck(l, ch_out, stride, preact):
    # stride is applied on the second conv, following fb.resnet.torch
    l, shortcut = apply_preactivation(l, preact)
    l = Conv2D('conv1', l, ch_out, 1, nl=BNReLU)
    l = Conv2D('conv2', l, ch_out, 3, stride=stride, nl=BNReLU)
    l = Conv2D('conv3', l, ch_out * 4, 1)
    return l + resnet_shortcut(shortcut, ch_out * 4, stride)


def preresnet_group(l, name, block_func, features, count, stride):
    with tf.variable_scope(name):
        for i in range(0, count):
            with tf.variable_scope('block{}'.format(i)):
                # first block doesn't need activation
                l = block_func(l, features,
                               stride if i == 0 else 1,
                               'no_preact' if i == 0 else 'bnrelu')
        # end of each group need an extra activation
        l = BNReLU('bnlast', l)
    return l


def resnet_basicblock(l, ch_out, stride):
    shortcut = l
    l = Conv2D('conv1', l, ch_out, 3, stride=stride, nl=BNReLU)
    l = Conv2D('conv2', l, ch_out, 3, nl=get_bn(zero_init=True))
    return l + resnet_shortcut(shortcut, ch_out, stride, nl=get_bn(zero_init=False))


def resnet_bottleneck_deeplab(l, ch_out, stride, dilation, stride_first=False):
    """
    stride_first: original resnet put stride on first conv. fb.resnet.torch put stride on second conv.
    """
    shortcut = l
    l = Conv2D('conv1', l, ch_out, 1, stride=stride if stride_first else 1, nl=BNReLU)
    if dilation == 1:
        l = Conv2D('conv2', l, ch_out, 3, stride=1 if stride_first else stride, nl=BNReLU)
    else:
        l = AtrousConv2D('conv2', l, ch_out, kernel_shape=3, rate=dilation, nl=BNReLU)
    l = Conv2D('conv3', l, ch_out * 4, 1, nl=get_bn(zero_init=True))
    return l + resnet_shortcut(shortcut, ch_out * 4, stride, nl=get_bn(zero_init=False))

def resnet_bottleneck(l, ch_out, stride, stride_first=False):
    """
    stride_first: original resnet put stride on first conv. fb.resnet.torch put stride on second conv.
    """
    shortcut = l
    l = Conv2D('conv1', l, ch_out, 1, stride=stride if stride_first else 1, nl=BNReLU)
    l = Conv2D('conv2', l, ch_out, 3, stride=1 if stride_first else stride, nl=BNReLU)
    l = Conv2D('conv3', l, ch_out * 4, 1, nl=get_bn(zero_init=True))
    return l + resnet_shortcut(shortcut, ch_out * 4, stride, nl=get_bn(zero_init=False))

def se_resnet_bottleneck(l, ch_out, stride):
    shortcut = l
    l = Conv2D('conv1', l, ch_out, 1, nl=BNReLU)
    l = Conv2D('conv2', l, ch_out, 3, stride=stride, nl=BNReLU)
    l = Conv2D('conv3', l, ch_out * 4, 1, nl=get_bn(zero_init=True))

    squeeze = GlobalAvgPooling('gap', l)
    squeeze = FullyConnected('fc1', squeeze, ch_out // 4, nl=tf.nn.relu)
    squeeze = FullyConnected('fc2', squeeze, ch_out * 4, nl=tf.nn.sigmoid)
    l = l * tf.reshape(squeeze, [-1, ch_out * 4, 1, 1])
    return l + resnet_shortcut(shortcut, ch_out * 4, stride, nl=get_bn(zero_init=False))


def resnet_group(l, name, block_func, features, count, stride, stride_first):
    with tf.variable_scope(name):
        for i in range(0, count):
            with tf.variable_scope('block{}'.format(i)):
                l = block_func(l, features, stride if i == 0 else 1, stride_first)
                # end of each block need an activation
                l = tf.nn.relu(l)
    return l


def fpn(resnet5,resnet4,resnet3,resnet2,resnet1,image,class_num):
    with tf.variable_scope("fpn"):
        with tf.variable_scope("res5"):
            resnet5 = resnet_basicblock(resnet5,class_num,stride=1)
            resnet5_upsample = tf.image.resize_bilinear(resnet5, resnet4.shape[1:3])
            with tf.variable_scope("smooth"):
                resnet5_upsample = resnet_basicblock(resnet5_upsample,class_num,stride=1)

        with tf.variable_scope("res4"):
            resnet4 =resnet_basicblock(resnet4,class_num,stride=1)
            resnet4 = resnet4 + resnet5_upsample
            resnet4_upsample = tf.image.resize_bilinear(resnet4, resnet3.shape[1:3])
            with tf.variable_scope("smooth"):
                resnet4_upsample = resnet_basicblock(resnet4_upsample, class_num, stride=1)
        with tf.variable_scope("res3"):
            resnet3 = resnet_basicblock(resnet3,class_num,stride=1)
            resnet3 = resnet3 + resnet4_upsample
            resnet3_upsample = tf.image.resize_bilinear(resnet3, resnet2.shape[1:3])
            with tf.variable_scope("smooth"):
                resnet3_upsample = resnet_basicblock(resnet3_upsample, class_num, stride=1)
        with tf.variable_scope("res2"):
            with tf.variable_scope("block1"):
                resnet2 = resnet_basicblock(resnet2,class_num,stride=1)
                resnet2 = resnet2 + resnet3_upsample
                resnet2_upsample = tf.image.resize_bilinear(resnet2, resnet1.shape[1:3])
            with tf.variable_scope("block2"):
                resnet2_upsample = resnet_basicblock(resnet2_upsample,class_num,stride=1)#smooth
                resnet2_upsample = tf.image.resize_bilinear(resnet2_upsample, image.shape[1:3])
            with tf.variable_scope("block3"):
                output = resnet_basicblock(resnet2_upsample, class_num, stride=1)#smooth

        return output

def resnet_backbone(image, num_blocks, group_func, block_func, class_num):
    with argscope(Conv2D, nl=tf.identity, use_bias=False,
                  W_init=variance_scaling_initializer(mode='FAN_OUT')):
        resnet1 = l = Conv2D('conv0', image, 64, 7, stride=2, nl=BNReLU)
        l = MaxPooling('pool0', l, shape=3, stride=2, padding='SAME')
        resnet2 = l = group_func(l, 'group0', block_func, 64, num_blocks[0], 1, stride_first=False)

        resnet3 = l = group_func(l, 'group1', block_func, 128, num_blocks[1], 2, stride_first=True)

        resnet4 = l = group_func(l, 'group2', block_func, 256, num_blocks[2], 2,  stride_first=True)

        resnet5 = group_func(l, 'group3', block_func, 512, num_blocks[3], 2,  stride_first=False)

    output = fpn(resnet5,resnet4,resnet3,resnet2,resnet1,image,class_num)

    return output