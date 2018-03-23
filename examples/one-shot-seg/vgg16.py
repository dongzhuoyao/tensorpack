#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: vgg16.py

import argparse
import os,cv2
import numpy as np

import tensorflow as tf

from tensorpack import *
from tensorpack.tfutils import argscope, get_model_loader
from tensorpack.tfutils.summary import *
from tensorpack.utils.gpu import get_nr_gpu

import OneShotDataset

max_epoch = 6
weight_decay = 5e-4
batch_size = 1
LR = 1e-10
CLASS_NUM = 2
support_image_size =(224,224)
query_image_size = (500, 500)

def get_data(name, batch=1):
    isTrain = True if 'train' in name else False
    dataset = OneShotDataset.OneShotDataset(name)
    dataset = BatchData(dataset, 1)
    return dataset


def convnormrelu(x, name, chan):
    x = Conv2D(name, x, chan, 3)
    if args.norm == 'bn':
        x = BatchNorm(name + '_bn', x)
    x = tf.nn.relu(x, name=name + '_relu')
    return x


def softmax_cross_entropy_with_ignore_label(logits, label, class_num):
    """
    This function accepts logits rather than predictions, and is more numerically stable than
    :func:`class_balanced_cross_entropy`.
    """
    with tf.name_scope('softmax_cross_entropy_with_ignore_label'):
        #tf.assert_equal(logits.shape[1], label.shape[1])  # shape assert
        #TODO need assert here
        raw_prediction = tf.reshape(logits, [-1, class_num])
        label = tf.reshape(label,[-1,])
        #label_onehot = tf.one_hot(label, depth=class_num)
        indices = tf.squeeze(tf.where(tf.less(label, class_num)), axis=1)
        #raw_gt = tf.reshape(label_onehot, [-1, class_num])

        gt = tf.gather(label, indices)
        prediction = tf.gather(raw_prediction, indices)

        # Pixel-wise softmax loss.
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=gt)
    return loss


class Model(ModelDesc):
    def inputs(self):
        return [tf.placeholder(tf.float32, [None, support_image_size[0], support_image_size[1], 3], 'first_image'),
                tf.placeholder(tf.int32, [None, support_image_size[0], support_image_size[1]], 'first_label'),
                tf.placeholder(tf.float32, [None, query_image_size[0], query_image_size[1], 3], 'second_image'),
                tf.placeholder(tf.int32, [None, query_image_size[0], query_image_size[1]], 'second_label')
                ]


    def build_graph(self, first_image, first_label, second_image, second_label):
        first_label = tf.expand_dims(first_label, 3, name='first_label')
        with argscope(Conv2D, kernel_size=3,
                      kernel_initializer=tf.variance_scaling_initializer(scale=2.)), \
             argscope([Conv2D, MaxPooling, BatchNorm], data_format="NHWC"):
                logits = (LinearWrap(first_image)
                      .apply(convnormrelu, 'conv1_1', 64)
                      .apply(convnormrelu, 'conv1_2', 64)
                      .MaxPooling('pool1', 2)
                      # 112
                      .apply(convnormrelu, 'conv2_1', 128)
                      .apply(convnormrelu, 'conv2_2', 128)
                      .MaxPooling('pool2', 2)
                      # 56
                      .apply(convnormrelu, 'conv3_1', 256)
                      .apply(convnormrelu, 'conv3_2', 256)
                      .apply(convnormrelu, 'conv3_3', 256)
                      .MaxPooling('pool3', 2)
                      # 28
                      .apply(convnormrelu, 'conv4_1', 512)
                      .apply(convnormrelu, 'conv4_2', 512)
                      .apply(convnormrelu, 'conv4_3', 512)
                      .MaxPooling('pool4', 2)
                      # 14
                      .apply(convnormrelu, 'conv5_1', 512)
                      .apply(convnormrelu, 'conv5_2', 512)
                      .apply(convnormrelu, 'conv5_3', 512)())

                # TODO smoothen
                logits = Conv2D("smooth", logits, CLASS_NUM, 3)
                logits = tf.image.resize_bilinear(logits, first_image.shape[1:3])

        costs = []

        cost = softmax_cross_entropy_with_ignore_label(logits, first_label, class_num=CLASS_NUM)
        cost = tf.reduce_mean(cost, name='cross_entropy_loss')
        costs.append(cost)

        if get_current_tower_context().is_training:
            wd_w = tf.train.exponential_decay(2e-4, get_global_step_var(),
                                              80000, 0.7, True)
            wd_cost = tf.multiply(wd_w, regularize_cost('.*/W', tf.nn.l2_loss), name='wd_cost')
            costs.append(wd_cost)

            add_param_summary(('.*', ['histogram', 'rms']))
            total_cost = tf.add_n(costs, name='cost')
            return total_cost

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=LR, trainable=False)
        opt = tf.train.AdamOptimizer(lr, epsilon=1e-3) #TODO,change to SGD
        return optimizer.apply_grad_processors(
            opt, [gradproc.ScaleGradient(
                [('nothing.*', 0.1), ('nothing.*', 5)])])



def get_config():
    nr_tower = max(get_nr_gpu(), 1)
    total_batch = batch_size * nr_tower


    logger.info("Running on {} towers. Batch size per tower: {}".format(nr_tower, batch_size))
    dataset_train = get_data('fold0_train', batch_size)
    dataset_val = get_data('fold0_1shot_test', batch_size)

    callbacks = [
        ModelSaver(),
        GPUUtilizationTracker(),
        EstimatedTimeLeft(),
        ProgressBar(["cross_entropy_loss", "cost", "wd_cost"])  # uncomment it to debug for every step
        #ScheduledHyperParamSetter(
        #    'learning_rate',
        #    [(0, 0.01), (3, max(BASE_LR, 0.01))], interp='linear'),
    ]

    input = QueueInput(dataset_train)
    input = StagingInput(input, nr_stage=1)
    return TrainConfig(
        model=Model(),
        data=input,
        callbacks=callbacks,
        steps_per_epoch=  10000 // total_batch,
        max_epoch=max_epoch,
    )


def view(args):
    ds = RepeatedData(get_data('fold0_train'), -1)
    ds.reset_state()
    for inputs in ds.get_data():
        ##"""
        cv2.imshow("first_img",(inputs[0][0]+np.array([104, 116, 122], dtype='float32')).astype(np.uint8))
        cv2.imshow("first_label",inputs[1][0])
        cv2.imshow("second_img", (inputs[2][0]+np.array([104, 116, 122], dtype='float32')).astype(np.uint8))
        cv2.imshow("second_label", inputs[3][0])
        cv2.waitKey(10000)
        ##"""
        print "ssss"
        pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='3',help='comma separated list of GPU(s) to use.')
    parser.add_argument('--data', help='ILSVRC dataset dir')
    parser.add_argument('--norm', choices=['none', 'bn'], default='none')
    parser.add_argument('--load',default="vgg16.npz", help='load model')
    parser.add_argument('--view', help='view dataset', action='store_true')
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.view:
        view(args)

    logger.set_logger_dir(os.path.join('train_log', 'vgg16-norm={}'.format(args.norm)))

    config = get_config()
    if args.load:
        config.session_init = get_model_loader(args.load)


    nr_tower = max(get_nr_gpu(), 1)
    trainer = SyncMultiGPUTrainerReplicated(nr_tower)
    launch_train_with_config(config, trainer)
