import numpy as np
import tensorflow as tf
import sys

from deeplab_resnet import model as deeplab101

from util.processing_tools import *



class RMI_model_onlydeeplab(object):

    def __init__(self, img, class_num ,batch_size,
                        mode,
                        vf_h = 40,
                        vf_w = 40,
                        H = 320,
                        W = 320,
                        keep_prob_rnn = 1.0,
                        keep_prob_emb = 1.0,
                        keep_prob_mlp = 1.0,
                        num_rnn_layers = 1,
                        weights = 'resnet'):
        self.batch_size = batch_size
        self.vf_h = vf_h
        self.vf_w = vf_w
        self.H = H
        self.W = W
        self.keep_prob_rnn = keep_prob_rnn
        self.keep_prob_emb = keep_prob_emb
        self.keep_prob_mlp = keep_prob_mlp
        self.num_rnn_layers = num_rnn_layers
        self.mode = mode
        self.weights = weights
        self.class_num = class_num

        self.im = img #tf.placeholder(tf.float32, [self.batch_size, self.H, self.W, 3])

        if self.weights == 'resnet':
            resmodel = resnet_model.ResNet(batch_size=self.batch_size, 
                                        atrous=True,
                                        images=self.im,
                                        labels=tf.constant(0.))
            self.visual_feat = resmodel.logits
        elif self.weights == 'deeplab':
            resmodel = deeplab101.DeepLabResNetModel({'data': self.im}, is_training=False)
            self.visual_feat = resmodel.layers['res5c_relu']

        with tf.variable_scope("mydeeplab"):
            self.build_graph()
            if self.mode == 'eval':
                return
            #self.train_op()

    def build_graph(self):

        if self.weights == 'deeplab':
            visual_feat = self._conv("conv0", self.visual_feat, 1, self.visual_feat.get_shape().as_list()[-1], 1000, [1, 1, 1, 1])
        elif self.weights == 'resnet':
            raise
            visual_feat  = self.visual_feat


        conv2 = self._conv("conv2", visual_feat, 1, visual_feat.get_shape().as_list()[-1], self.class_num, [1, 1, 1, 1])
        self.pred = conv2
        self.up = tf.image.resize_bilinear(self.pred, [self.H, self.W])
        self.sigm = tf.sigmoid(self.up)


    def _conv(self, name, x, filter_size, in_filters, out_filters, strides):
        with tf.variable_scope(name):
            w = tf.get_variable('DW', [filter_size, filter_size, in_filters, out_filters], 
                initializer=tf.contrib.layers.xavier_initializer_conv2d())
            b = tf.get_variable('biases', out_filters, initializer=tf.constant_initializer(0.))
            return tf.nn.conv2d(x, w, strides, padding='SAME') + b

    def _atrous_conv(self, name, x, filter_size, in_filters, out_filters, rate):
        with tf.variable_scope(name):
            w = tf.get_variable('DW', [filter_size, filter_size, in_filters, out_filters],
                initializer=tf.random_normal_initializer(stddev=0.01))
            b = tf.get_variable('biases', out_filters, initializer=tf.constant_initializer(0.))
            return tf.nn.atrous_conv2d(x, w, rate=rate, padding='SAME') + b