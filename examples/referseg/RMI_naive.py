import numpy as np
import tensorflow as tf
import sys
# sys.path.append('./external/TF-resnet')
# sys.path.append('./external/TF-deeplab')
# import resnet_model
# import deeplab_model
from deeplab_resnet import model as deeplab101
from util.cell import ConvLSTMCell

from util import data_reader
from util.processing_tools import *
from util import im_processing, text_processing, eval_tools
from util import loss


class LSTM_model(object):

    def __init__(self,  im, words, vocab_size, num_steps, batch_size, class_num,
                        vf_h = 40,
                        vf_w = 40,
                        H = 320,
                        W = 320,
                        vf_dim = 2048,
                        w_emb_dim = 1000,
                        v_emb_dim = 1000,
                        rnn_size = 1000,
                        keep_prob_rnn = 1.0,
                        keep_prob_emb = 1.0,
                        keep_prob_mlp = 1.0,
                        num_rnn_layers = 1,
                        mode = 'eval',
                        weights = 'resnet',
                        conv5 = False):
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.class_num = class_num
        self.vf_h = vf_h
        self.vf_w = vf_w
        self.H = H
        self.W = W
        self.vf_dim = vf_dim
        self.vocab_size = vocab_size
        self.w_emb_dim = w_emb_dim
        self.v_emb_dim = v_emb_dim
        self.rnn_size = rnn_size
        self.keep_prob_rnn = keep_prob_rnn
        self.keep_prob_emb = keep_prob_emb
        self.keep_prob_mlp = keep_prob_mlp
        self.num_rnn_layers = num_rnn_layers
        self.mode = mode
        self.weights = weights
        self.conv5 = conv5

        self.words = words#tf.placeholder(tf.int32, [self.batch_size, self.num_steps])
        self.im = im #tf.placeholder(tf.float32, [self.batch_size, self.H, self.W, 3])


        if self.weights == 'resnet':
            resmodel = resnet_model.ResNet(batch_size=self.batch_size, 
                                        atrous=True,
                                        images=self.im,
                                        labels=tf.constant(0.))
            self.visual_feat = resmodel.logits
            
        elif self.weights == 'deeplab':
            # resmodel = deeplab_model.DeepLab(batch_size=self.batch_size,
            #                             images=self.im,
            #                             labels=tf.constant(0.))
            # self.visual_feat = resmodel.res5c
            resmodel = deeplab101.DeepLabResNetModel({'data': self.im}, is_training=False)
            self.visual_feat = resmodel.layers['res5c_relu']
            self.visual_feat_c4 = resmodel.layers['res4b22_relu']
            self.visual_feat_c3 = resmodel.layers['res3b3_relu']

        with tf.variable_scope("text_objseg"):
            self.build_graph()
            if self.mode == 'eval':
                return
            #self.train_op()

    def build_graph(self):

        if self.weights == 'deeplab':
            # atrous0 = self._atrous_conv("atrous0", self.visual_feat, 3, self.vf_dim, self.v_emb_dim, 6)
            # atrous1 = self._atrous_conv("atrous1", self.visual_feat, 3, self.vf_dim, self.v_emb_dim, 12)
            # atrous2 = self._atrous_conv("atrous2", self.visual_feat, 3, self.vf_dim, self.v_emb_dim, 18)
            # atrous3 = self._atrous_conv("atrous3", self.visual_feat, 3, self.vf_dim, self.v_emb_dim, 24)
            # visual_feat = tf.add(atrous0, atrous1)
            # visual_feat = tf.add(visual_feat, atrous2)
            # visual_feat = tf.add(visual_feat, atrous3)
            visual_feat = self._conv("mlp0", self.visual_feat, 1, self.vf_dim, self.v_emb_dim, [1, 1, 1, 1])
        elif self.weights == 'resnet':
            visual_feat = self.visual_feat
            
        embedding_mat = tf.get_variable("embedding", [self.vocab_size, self.w_emb_dim], 
                                        initializer=tf.random_uniform_initializer(minval=-0.08, maxval=0.08))
        embedded_seq = tf.nn.embedding_lookup(embedding_mat, tf.transpose(self.words)) #[step, batch_size, emb_dim]

        rnn_cell_basic = tf.nn.rnn_cell.BasicLSTMCell(self.rnn_size, state_is_tuple=False)

        if self.mode == 'train' and self.keep_prob_rnn < 1:
            rnn_cell_basic = tf.nn.rnn_cell.DropoutWrapper(rnn_cell_basic, output_keep_prob=self.keep_prob_rnn)
        cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell_basic] * self.num_rnn_layers, state_is_tuple=False)

        state = cell.zero_state(self.batch_size, tf.float32)
        state_shape = state.get_shape().as_list()
        state_shape[0] = self.batch_size
        state.set_shape(state_shape) # [batch_size, emb_dim]

        def f1():
            return tf.constant(0.), state

        def f2():
            # Word input to embedding layer
            w_emb = embedded_seq[n, :, :]
            if self.mode == 'train' and self.keep_prob_emb < 1:
                w_emb = tf.nn.dropout(w_emb, self.keep_prob_emb)
            return cell(w_emb, state)

        """
        with tf.variable_scope("RNN"):
            for n in range(self.num_steps):
                if n > 0:
                    tf.get_variable_scope().reuse_variables()
                #rnn_output, state = cell(w_emb, state)
                rnn_output, state = tf.cond(tf.equal(self.words[0, n], tf.constant(0)), f1, f2) # ??? 0-th batch?
        """

        with tf.variable_scope("RNN"):
            for n in range(self.num_steps):
                if n > 0:
                    tf.get_variable_scope().reuse_variables()
                w_emb = embedded_seq[n, :, :]
                rnn_output, state = cell(w_emb, state)


        lang_feat = tf.reshape(rnn_output, [self.batch_size, 1, 1, self.rnn_size])
        lang_feat = tf.nn.l2_normalize(lang_feat, 3)
        lang_feat = tf.tile(lang_feat, [1, self.vf_h, self.vf_w, 1])

        # Generate spatial grid
        visual_feat = tf.nn.l2_normalize(visual_feat, 3)
        spatial = tf.convert_to_tensor(generate_spatial_batch(self.batch_size, self.vf_h, self.vf_w))

        feat_all = tf.concat([visual_feat, lang_feat, spatial], 3)

        self.pred = self._conv("score", feat_all, 1, feat_all.get_shape().as_list()[-1], self.class_num, [1, 1, 1, 1])

        self.up = tf.image.resize_bilinear(self.pred, [self.H, self.W]) # final feature map
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
