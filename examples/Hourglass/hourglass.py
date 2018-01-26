#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: hed.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import cv2
import tensorflow as tf
import argparse
from six.moves import zip
import os


from tensorpack import *
from tensorpack.dataflow import dataset
from tensorpack.utils.gpu import get_nr_gpu
from tensorpack.tfutils import optimizer
from tensorpack.tfutils.summary import add_moving_summary, add_param_summary
from tensorpack.utils import logger
import numpy as np

image_size =(256,256)
heatmap_size = (64,64)

class hg():
    def __init__(self,train):
        self.nFeat =256
        self.nLow = 4
        self.nFeat = 4
        self.train = train

    def _graph_hourglass(self, inputs):
        """Create the Network
        Args:
            inputs : TF Tensor (placeholder) of shape (None, 256, 256, 3) #TODO : Create a parameter for customize size
        """
        with tf.name_scope('model'):
            with tf.name_scope('preprocessing'):
                # Input Dim : nbImages x 256 x 256 x 3
                pad1 = tf.pad(inputs, [[0, 0], [2, 2], [2, 2], [0, 0]], name='pad_1')
                # Dim pad1 : nbImages x 260 x 260 x 3
                conv1 = self._conv_bn_relu(pad1, filters=64, kernel_size=6, strides=2, name='conv_256_to_128')
                # Dim conv1 : nbImages x 128 x 128 x 64
                r1 = self._residual(conv1, numOut=128, name='r1')
                # Dim pad1 : nbImages x 128 x 128 x 128
                pool1 = tf.contrib.layers.max_pool2d(r1, [2, 2], [2, 2], padding='VALID')
                # Dim pool1 : nbImages x 64 x 64 x 128
                if self.tiny:
                    r3 = self._residual(pool1, numOut=self.nFeat, name='r3')
                else:
                    r2 = self._residual(pool1, numOut=int(self.nFeat / 2), name='r2')
                    r3 = self._residual(r2, numOut=self.nFeat, name='r3')
            # Storage Table
            hg = [None] * self.nStack
            ll = [None] * self.nStack
            ll_ = [None] * self.nStack
            drop = [None] * self.nStack
            out = [None] * self.nStack
            out_ = [None] * self.nStack
            sum_ = [None] * self.nStack
            if self.tiny:
                with tf.name_scope('stacks'):
                    with tf.name_scope('stage_0'):
                        hg[0] = self._hourglass(r3, self.nLow, self.nFeat, 'hourglass')
                        drop[0] = tf.layers.dropout(hg[0], rate=self.dropout_rate, training=self.training,
                                                    name='dropout')
                        ll[0] = self._conv_bn_relu(drop[0], self.nFeat, 1, 1, name='ll')
                        if self.modif:
                            # TEST OF BATCH RELU
                            out[0] = self._conv_bn_relu(ll[0], self.outDim, 1, 1, 'VALID', 'out')
                        else:
                            out[0] = self._conv(ll[0], self.outDim, 1, 1, 'VALID', 'out')
                        out_[0] = self._conv(out[0], self.nFeat, 1, 1, 'VALID', 'out_')
                        sum_[0] = tf.add_n([out_[0], ll[0], r3], name='merge')
                    for i in range(1, self.nStack - 1):
                        with tf.name_scope('stage_' + str(i)):
                            hg[i] = self._hourglass(sum_[i - 1], self.nLow, self.nFeat, 'hourglass')
                            drop[i] = tf.layers.dropout(hg[i], rate=self.dropout_rate, training=self.training,
                                                        name='dropout')
                            ll[i] = self._conv_bn_relu(drop[i], self.nFeat, 1, 1, name='ll')
                            if self.modif:
                                # TEST OF BATCH RELU
                                out[i] = self._conv_bn_relu(ll[i], self.outDim, 1, 1, 'VALID', 'out')
                            else:
                                out[i] = self._conv(ll[i], self.outDim, 1, 1, 'VALID', 'out')
                            out_[i] = self._conv(out[i], self.nFeat, 1, 1, 'VALID', 'out_')
                            sum_[i] = tf.add_n([out_[i], ll[i], sum_[i - 1]], name='merge')
                    with tf.name_scope('stage_' + str(self.nStack - 1)):
                        hg[self.nStack - 1] = self._hourglass(sum_[self.nStack - 2], self.nLow, self.nFeat, 'hourglass')
                        drop[self.nStack - 1] = tf.layers.dropout(hg[self.nStack - 1], rate=self.dropout_rate,
                                                                  training=self.training, name='dropout')
                        ll[self.nStack - 1] = self._conv_bn_relu(drop[self.nStack - 1], self.nFeat, 1, 1, 'VALID',
                                                                 'conv')
                        if self.modif:
                            out[self.nStack - 1] = self._conv_bn_relu(ll[self.nStack - 1], self.outDim, 1, 1, 'VALID',
                                                                      'out')
                        else:
                            out[self.nStack - 1] = self._conv(ll[self.nStack - 1], self.outDim, 1, 1, 'VALID', 'out')
                if self.modif:
                    return tf.nn.sigmoid(tf.stack(out, axis=1, name='stack_output'), name='final_output')
                else:
                    return tf.stack(out, axis=1, name='final_output')
            else:
                with tf.name_scope('stacks'):
                    with tf.name_scope('stage_0'):
                        hg[0] = self._hourglass(r3, self.nLow, self.nFeat, 'hourglass')
                        drop[0] = tf.layers.dropout(hg[0], rate=self.dropout_rate, training=self.training,
                                                    name='dropout')
                        ll[0] = self._conv_bn_relu(drop[0], self.nFeat, 1, 1, 'VALID', name='conv')
                        ll_[0] = self._conv(ll[0], self.nFeat, 1, 1, 'VALID', 'll')
                        if self.modif:
                            # TEST OF BATCH RELU
                            out[0] = self._conv_bn_relu(ll[0], self.outDim, 1, 1, 'VALID', 'out')
                        else:
                            out[0] = self._conv(ll[0], self.outDim, 1, 1, 'VALID', 'out')
                        out_[0] = self._conv(out[0], self.nFeat, 1, 1, 'VALID', 'out_')
                        sum_[0] = tf.add_n([out_[0], r3, ll_[0]], name='merge')
                    for i in range(1, self.nStack - 1):
                        with tf.name_scope('stage_' + str(i)):
                            hg[i] = self._hourglass(sum_[i - 1], self.nLow, self.nFeat, 'hourglass')
                            drop[i] = tf.layers.dropout(hg[i], rate=self.dropout_rate, training=self.training,
                                                        name='dropout')
                            ll[i] = self._conv_bn_relu(drop[i], self.nFeat, 1, 1, 'VALID', name='conv')
                            ll_[i] = self._conv(ll[i], self.nFeat, 1, 1, 'VALID', 'll')
                            if self.modif:
                                out[i] = self._conv_bn_relu(ll[i], self.outDim, 1, 1, 'VALID', 'out')
                            else:
                                out[i] = self._conv(ll[i], self.outDim, 1, 1, 'VALID', 'out')
                            out_[i] = self._conv(out[i], self.nFeat, 1, 1, 'VALID', 'out_')
                            sum_[i] = tf.add_n([out_[i], sum_[i - 1], ll_[0]], name='merge')
                    with tf.name_scope('stage_' + str(self.nStack - 1)):
                        hg[self.nStack - 1] = self._hourglass(sum_[self.nStack - 2], self.nLow, self.nFeat, 'hourglass')
                        drop[self.nStack - 1] = tf.layers.dropout(hg[self.nStack - 1], rate=self.dropout_rate,
                                                                  training=self.training, name='dropout')
                        ll[self.nStack - 1] = self._conv_bn_relu(drop[self.nStack - 1], self.nFeat, 1, 1, 'VALID',
                                                                 'conv')
                        if self.modif:
                            out[self.nStack - 1] = self._conv_bn_relu(ll[self.nStack - 1], self.outDim, 1, 1, 'VALID',
                                                                      'out')
                        else:
                            out[self.nStack - 1] = self._conv(ll[self.nStack - 1], self.outDim, 1, 1, 'VALID', 'out')
                if self.modif:
                    return tf.nn.sigmoid(tf.stack(out, axis=1, name='stack_output'), name='final_output')
                else:
                    return tf.stack(out, axis=1, name='final_output')

    def _conv(self, inputs, filters, kernel_size=1, strides=1, pad='VALID', name='conv'):
        with tf.name_scope(name):
            # Kernel for convolution, Xavier Initialisation
            kernel = tf.Variable(tf.contrib.layers.xavier_initializer(uniform=False)(
                [kernel_size, kernel_size, inputs.get_shape().as_list()[3], filters]), name='weights')
            conv = tf.nn.conv2d(inputs, kernel, [1, strides, strides, 1], padding=pad, data_format='NHWC')
            return conv

    def _conv_bn_relu(self, inputs, filters, kernel_size=1, strides=1, pad='VALID', name='conv_bn_relu'):
        with tf.name_scope(name):
            kernel = tf.Variable(tf.contrib.layers.xavier_initializer(uniform=False)(
                [kernel_size, kernel_size, inputs.get_shape().as_list()[3], filters]), name='weights')
            conv = tf.nn.conv2d(inputs, kernel, [1, strides, strides, 1], padding='VALID', data_format='NHWC')
            norm = tf.contrib.layers.batch_norm(conv, 0.9, epsilon=1e-5, activation_fn=tf.nn.relu,
                                                is_training=self.training)
            return norm

    def _conv_block(self, inputs, numOut, name='conv_block'):
        if self.tiny:
            with tf.name_scope(name):
                norm = tf.contrib.layers.batch_norm(inputs, 0.9, epsilon=1e-5, activation_fn=tf.nn.relu,
                                                    is_training=self.training)
                pad = tf.pad(norm, np.array([[0, 0], [1, 1], [1, 1], [0, 0]]), name='pad')
                conv = self._conv(pad, int(numOut), kernel_size=3, strides=1, pad='VALID', name='conv')
                return conv
        else:
            with tf.name_scope(name):
                with tf.name_scope('norm_1'):
                    norm_1 = tf.contrib.layers.batch_norm(inputs, 0.9, epsilon=1e-5, activation_fn=tf.nn.relu,
                                                          is_training=self.training)
                    conv_1 = self._conv(norm_1, int(numOut / 2), kernel_size=1, strides=1, pad='VALID', name='conv')
                with tf.name_scope('norm_2'):
                    norm_2 = tf.contrib.layers.batch_norm(conv_1, 0.9, epsilon=1e-5, activation_fn=tf.nn.relu,
                                                          is_training=self.training)
                    pad = tf.pad(norm_2, np.array([[0, 0], [1, 1], [1, 1], [0, 0]]), name='pad')
                    conv_2 = self._conv(pad, int(numOut / 2), kernel_size=3, strides=1, pad='VALID', name='conv')
                with tf.name_scope('norm_3'):
                    norm_3 = tf.contrib.layers.batch_norm(conv_2, 0.9, epsilon=1e-5, activation_fn=tf.nn.relu,
                                                          is_training=self.training)
                    conv_3 = self._conv(norm_3, int(numOut), kernel_size=1, strides=1, pad='VALID', name='conv')
                return conv_3

    def _skip_layer(self, inputs, numOut, name='skip_layer'):
        with tf.name_scope(name):
            if inputs.get_shape().as_list()[3] == numOut:
                return inputs
            else:
                conv = self._conv(inputs, numOut, kernel_size=1, strides=1, name='conv')
                return conv

    def _residual(self, inputs, numOut, name='residual_block'):
        with tf.name_scope(name):
            convb = self._conv_block(inputs, numOut)
            skipl = self._skip_layer(inputs, numOut)
            if self.modif:
                return tf.nn.relu(tf.add_n([convb, skipl], name='res_block'))
            else:
                return tf.add_n([convb, skipl], name='res_block')

    def _hourglass(self, inputs, n, numOut, name='hourglass'):
        with tf.name_scope(name):
            # Upper Branch
            up_1 = self._residual(inputs, numOut, name='up_1')
            # Lower Branch
            low_ = tf.contrib.layers.max_pool2d(inputs, [2, 2], [2, 2], padding='VALID')
            low_1 = self._residual(low_, numOut, name='low_1')

            if n > 0:
                low_2 = self._hourglass(low_1, n - 1, numOut, name='low_2')
            else:
                low_2 = self._residual(low_1, numOut, name='low_2')

            low_3 = self._residual(low_2, numOut, name='low_3')
            up_2 = tf.image.resize_nearest_neighbor(low_3, tf.shape(low_3)[1:3] * 2, name='upsampling')
            if self.modif:
                # Use of RELU
                return tf.nn.relu(tf.add_n([up_2, up_1]), name='out_hg')
            else:
                return tf.add_n([up_2, up_1], name='out_hg')

class Model(ModelDesc):
    def _get_inputs(self):
        return [InputDesc(tf.float32, [None, image_size[0], image_size[1], 3], 'image'),
                InputDesc(tf.float32, [None, heatmap_size[0], heatmap_size[1]], 'heatmap')]

    def _build_graph(self, inputs):
        image, edgemap = inputs
        image = image - tf.constant([104, 116, 122], dtype='float32')
        edgemap = tf.expand_dims(edgemap, 3, name='edgemap4d')

        ctx = get_current_tower_context()
        logger.info("current ctx.is_training: {}".format(ctx.is_training))
        m = hg(ctx.is_training)
        pred = m._graph_hourglass(image)

        costs =[]
        costs.append(tf.losses.mean_squared_error(edgemap, pred))

        if get_current_tower_context().is_training:
            wd_w = tf.train.exponential_decay(2e-4, get_global_step_var(),
                                              80000, 0.7, True)
            wd_cost = tf.multiply(wd_w, regularize_cost('.*/W', tf.nn.l2_loss), name='wd_cost')
            costs.append(wd_cost)
            self.cost = tf.add_n(costs, name='cost')


    def _get_optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=3e-5, trainable=False)
        opt = tf.train.AdamOptimizer(lr, epsilon=1e-3)
        return optimizer.apply_grad_processors(
            opt, [gradproc.ScaleGradient(
                [('convfcweight.*', 0.1), ('conv5_.*', 5)])])


def get_data(name):
    isTrain = name == 'train'
    ds = dataset.mpii(name, "metadata/mpii_annotations.json", shuffle=True)



    if isTrain:
        shape_aug = [
            imgaug.RandomResize(xrange=(0.7, 1.5), yrange=(0.7, 1.5),
                                aspect_ratio_thres=0.15),
            imgaug.RotationAndCropValid(90),
            imgaug.Flip(horiz=True),
            imgaug.Flip(vert=True)
        ]
    else:
        # the original image shape (321x481) in BSDS is not a multiple of 16
        IMAGE_SHAPE = (320, 480)
        shape_aug = [imgaug.CenterCrop(IMAGE_SHAPE)]
    ds = AugmentImageComponents(ds, shape_aug, (0, 1), copy=False)

    def f(m):   # thresholding
        m[m >= 0.50] = 1
        m[m < 0.50] = 0
        return m
    ds = MapDataComponent(ds, f, 1)

    if isTrain:
        augmentors = [
            imgaug.Brightness(63, clip=False),
            imgaug.Contrast((0.4, 1.5)),
        ]
        ds = AugmentImageComponent(ds, augmentors, copy=False)
        ds = BatchDataByShape(ds, 8, idx=0)
        ds = PrefetchDataZMQ(ds, 1)
    else:
        ds = BatchData(ds, 1)
    return ds


def view_data():
    ds = RepeatedData(get_data('train'), -1)
    ds.reset_state()
    for ims, edgemaps in ds.get_data():
        for im, edgemap in zip(ims, edgemaps):
            assert im.shape[0] % 16 == 0 and im.shape[1] % 16 == 0, im.shape
            cv2.imshow("im", im / 255.0)
            cv2.waitKey(1000)
            cv2.imshow("edge", edgemap)
            cv2.waitKey(1000)


def get_config():
    logger.auto_set_dir()
    dataset_train = get_data('train')
    steps_per_epoch = dataset_train.size() * 40
    dataset_val = get_data('val')

    return TrainConfig(
        dataflow=dataset_train,
        callbacks=[
            ModelSaver(),
            ScheduledHyperParamSetter('learning_rate', [(30, 6e-6), (45, 1e-6), (60, 8e-7)]),
            HumanHyperParamSetter('learning_rate'),
            InferenceRunner(dataset_val,
                            BinaryClassificationStats('prediction', 'edgemap4d'))
        ],
        model=Model(),
        steps_per_epoch=steps_per_epoch,
        max_epoch=100,
    )


def run(model_path, image_path, output):
    pred_config = PredictConfig(
        model=Model(),
        session_init=get_model_loader(model_path),
        input_names=['image'],
        output_names=['output' + str(k) for k in range(1, 7)])
    predictor = OfflinePredictor(pred_config)
    im = cv2.imread(image_path)
    assert im is not None
    im = cv2.resize(
        im, (im.shape[1] // 16 * 16, im.shape[0] // 16 * 16)
    )[None, :, :, :].astype('float32')
    outputs = predictor(im)
    if output is None:
        for k in range(6):
            pred = outputs[k][0]
            cv2.imwrite("out{}.png".format(
                '-fused' if k == 5 else str(k + 1)), pred * 255)
    else:
        pred = outputs[5][0]
        cv2.imwrite(output, pred * 255)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--view', help='view dataset', action='store_true')
    parser.add_argument('--run', help='run model on images')
    parser.add_argument('--output', help='fused output filename. default to out-fused.png')
    args = parser.parse_args()
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.view:
        view_data()
    elif args.run:
        run(args.load, args.run, args.output)
    else:
        config = get_config()
        if args.load:
            config.session_init = get_model_loader(args.load)
        launch_train_with_config(
            config,
            SyncMultiGPUTrainer(max(get_nr_gpu(), 1)))
