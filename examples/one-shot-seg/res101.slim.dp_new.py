#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: vgg16.py

import argparse
import os,cv2
import numpy as np

from tqdm import tqdm
from tensorpack import *
from tensorpack.tfutils import argscope, get_model_loader
from tensorpack.utils.segmentation.segmentation import predict_slider, visualize_label, predict_scaler
from tensorpack.tfutils.summary import *
from tensorpack.utils.gpu import get_nr_gpu
from tensorpack.utils.stats import MIoUStatistics
import OneShotDataset
from deeplabv2_dilation6_new import deeplabv2
import tensorflow as tf
slim = tf.contrib.slim


max_epoch = 6
weight_decay = 5e-4
batch_size = 12
LR = 1e-4
CLASS_NUM = 2
evaluate_every_n_epoch = 1
support_image_size =(321, 321)
query_image_size = (321, 321)
n_times_per_image_in_a_epoch = 10000

def get_data(name,batch_size=1):
    isTrain = True if 'train' in name else False
    dataset = OneShotDataset.OneShotDataset(name)
    dataset = BatchData(dataset, batch_size)
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
        tf.assert_equal(logits.get_shape()[1:3], label.get_shape()[1:])  # shape assert

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
        return [#tf.placeholder(tf.float32, [None, support_image_size[0], support_image_size[1], 3], 'first_image'),
                #tf.placeholder(tf.float32, [None, support_image_size[0], support_image_size[1]], 'first_label'),
                tf.placeholder(tf.float32, [None, query_image_size[0], query_image_size[1], 3], 'second_image'),
                tf.placeholder(tf.int32, [None, query_image_size[0], query_image_size[1]], 'second_label')
                ]



    def build_graph(self, second_image, second_label):
        ctx = get_current_tower_context()
        logger.info("current ctx.is_training: {}".format(ctx.is_training))

        #with tf.variable_scope("support"):
        #    support_logits = deeplabv2(first_image_masked, CLASS_NUM, is_training=ctx.is_training)
        with tf.variable_scope("query"):
            query_logits = deeplabv2(second_image, CLASS_NUM, is_training=ctx.is_training)

        costs = []
        logits = query_logits

        logits = slim.conv2d(logits, CLASS_NUM, [3, 3], stride=1, rate=6,
                          activation_fn=None, normalizer_fn=None)

        logits = tf.image.resize_bilinear(logits, second_image.shape[1:3],name="upsample")

        prob = tf.nn.softmax(logits, name='prob')



        if get_current_tower_context().is_training:
            cost = softmax_cross_entropy_with_ignore_label(logits, second_label, class_num=CLASS_NUM)
            cost = tf.reduce_mean(cost, name='cross_entropy_loss')
            costs.append(cost)

            wd_w = tf.train.exponential_decay(2e-4, get_global_step_var(),
                                              80000, 0.7, True)
            wd_cost = tf.multiply(wd_w, regularize_cost('.*/W', tf.nn.l2_loss), name='wd_cost')
            costs.append(wd_cost)

            #add_param_summary(('.*', ['histogram', 'rms']))
            total_cost = tf.add_n(costs, name='cost')
            return total_cost

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=LR, trainable=False)
        #opt = tf.train.AdamOptimizer(lr, epsilon=1e-3) #TODO,change to SGD
        opt = tf.train.MomentumOptimizer(lr,momentum=0.99)
        return optimizer.apply_grad_processors(
            opt, [gradproc.ScaleGradient(
                [('nothing.*', 0.1), ('nothing.*', 5)])])



def get_config():
    logger.auto_set_dir()
    nr_tower = max(get_nr_gpu(), 1)
    total_batch = batch_size * nr_tower

    logger.info("Running on {} towers. Batch size per tower: {}".format(nr_tower, batch_size))
    dataset_train = get_data('fold0_train', batch_size)

    callbacks = [
        ModelSaver(),
        GPUUtilizationTracker(),
        EstimatedTimeLeft(),
        PeriodicTrigger(CalculateMIoU(CLASS_NUM), every_k_epochs=evaluate_every_n_epoch),
        ProgressBar(["cross_entropy_loss", "cost", "wd_cost"]) , # uncomment it to debug for every step
        #RunOp(lambda: tf.add_check_numerics_ops(), run_before=False, run_as_trigger=True, run_step=True)

    ]

    input = QueueInput(dataset_train)
    input = StagingInput(input, nr_stage=1)
    return TrainConfig(
        model=Model(),
        data=input,
        callbacks=callbacks,
        steps_per_epoch=  n_times_per_image_in_a_epoch // total_batch,
        max_epoch=max_epoch,
    )


class CalculateMIoU(Callback):
    def __init__(self, nb_class):
        self.nb_class = nb_class

    def _setup_graph(self):
        self.pred = self.trainer.get_predictor(
            ['second_image'], ['prob'])

    def _before_train(self):
        pass

    def _trigger(self):
        global args
        self.val_ds = get_data('fold0_1shot_test')
        self.val_ds.reset_state()

        self.stat = MIoUStatistics(self.nb_class)

        for second_image, second_label in tqdm(self.val_ds.get_data()):
            #first_image = np.squeeze(first_image)
            #first_label = np.squeeze(first_label)
            second_image = np.squeeze(second_image)
            second_label = np.squeeze(second_label)

            def mypredictor(input_img):
                # input image: 1*H*W*3
                # output : H*W*C
                output = self.pred(input_img)
                return output[0][0]
            prediction = predict_scaler(second_image, mypredictor, scales=[0.5,0.75, 1, 1.25, 1.5], classes=CLASS_NUM, tile_size=query_image_size,
                           is_densecrf=False)
            prediction = np.argmax(prediction, axis=2)
            self.stat.feed(prediction, second_label)

        self.trainer.monitors.put_scalar("mIoU", self.stat.mIoU)
        self.trainer.monitors.put_scalar("mean_accuracy", self.stat.mean_accuracy)
        self.trainer.monitors.put_scalar("accuracy", self.stat.accuracy)


def view(args):
    ds = RepeatedData(get_data('fold0_train'), -1)
    ds.reset_state()
    for inputs in ds.get_data():
        ##"""
        cv2.imshow("first_img",(inputs[0][0]+np.array([104, 116, 122], dtype='float32')).astype(np.uint8))
        cv2.imshow("first_label",inputs[1][0])
        #cv2.imshow("second_img", (inputs[2][0]+np.array([104, 116, 122], dtype='float32')).astype(np.uint8))
        #cv2.imshow("second_label", inputs[3][0])
        cv2.waitKey(10000)
        ##"""
        print "ssss"
        pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='4',help='comma separated list of GPU(s) to use.')
    parser.add_argument('--data', help='ILSVRC dataset dir')
    parser.add_argument('--norm', choices=['none', 'bn'], default='none')
    parser.add_argument('--load',default="slim_resnet_v2_101.ckpt", help='load model')
    parser.add_argument('--view', help='view dataset', action='store_true')
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.view:
        view(args)


    config = get_config()
    if args.load:
        from sess_utils import my_get_model_loader
        config.session_init = my_get_model_loader(args.load)


    nr_tower = max(get_nr_gpu(), 1)
    trainer = SyncMultiGPUTrainerReplicated(nr_tower)
    launch_train_with_config(config, trainer)
