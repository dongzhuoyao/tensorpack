#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: deeplabFOV.py
# Author: Tao Hu <taohu620@gmail.com>

import cv2
import tensorflow as tf
import argparse
from six.moves import zip
import os
import numpy as np

os.environ['TENSORPACK_TRAIN_API'] = 'v2'   # will become default soon
from tensorpack import *
from tensorpack.dataflow import dataset
from tensorpack.utils.gpu import get_nr_gpu
from tensorpack.utils.segmentation.segmentation import predict_slider, visualize_label, predict_scaler
from tensorpack.utils.stats import MIoUStatistics
from tensorpack.utils import logger
from tensorpack.tfutils import optimizer
from tensorpack.tfutils.summary import add_moving_summary, add_param_summary
from tqdm import tqdm
from RMI_model import RMI_model
from data_loader import DataLoader

CLASS_NUM = DataLoader.class_num()
IMG_SIZE = 320
IGNORE_LABEL = 255
MAX_LENGTH = 49
VOCAB_SIZE = len(DataLoader(name = "train", max_length=MAX_LENGTH, img_size=IMG_SIZE).word_to_idx.keys())#3224#28645#24022  # careful about the VOCAB SIZE
# maximum length of caption(number of word). if caption is longer than max_length, deleted.
STEP_NUM = MAX_LENGTH+2 # equal Max Length
evaluate_every_n_epoch = 1
max_epoch = 10
init_lr = 2.5e-4
lr_schedule = [(3, 1e-4), (7, 1e-5)]
step_scale = 6

def softmax_cross_entropy_with_ignore_label(logits, label, class_num):
    """
    This function accepts logits rather than predictions, and is more numerically stable than
    :func:`class_balanced_cross_entropy`.
    """
    with tf.name_scope('softmax_cross_entropy_with_ignore_label'):
        tf.assert_equal(logits.get_shape()[1:3], label.get_shape()[1:3])  # shape assert

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

    def _get_inputs(self):
        return [InputDesc(tf.float32, [None, IMG_SIZE, IMG_SIZE, 3], 'image'),
                InputDesc(tf.int32, [None, IMG_SIZE, IMG_SIZE, 1], 'gt'),
                InputDesc(tf.int32, [None, STEP_NUM], 'caption'),]

    def _build_graph(self, inputs):
        image, label, caption = inputs
        image = image - tf.constant([104, 116, 122], dtype='float32')
        mode = "train" if get_current_tower_context().is_training else "val"
        current_batch_size = args.batch_size if get_current_tower_context().is_training else 1
        model = RMI_model(image, caption, class_num=CLASS_NUM, batch_size=current_batch_size, num_steps= STEP_NUM, mode=mode, vocab_size=VOCAB_SIZE, weights="deeplab")
        predict = model.up

        label = tf.identity(label, name="label")


        costs = []
        prob = tf.identity(predict, name='prob')
        prediction = tf.argmax(prob, axis=-1, name="prediction")

        cost = softmax_cross_entropy_with_ignore_label(logits=prob, label=label,
                                                             class_num=CLASS_NUM)

        cost = tf.reduce_mean(cost, name='cross_entropy_loss')  # the average cross-entropy loss
        costs.append(cost)

        if get_current_tower_context().is_training:
            wd_w = tf.train.exponential_decay(2e-4, get_global_step_var(),
                                              80000, 0.7, True)
            wd_cost = tf.multiply(wd_w, regularize_cost('.*/Waaaaaa', tf.nn.l2_loss), name='wd_cost') #TODO
            #wd_cost = 0.0
            costs.append(wd_cost)
            self.cost = tf.add_n(costs, name='cost')


    def _get_optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=init_lr, trainable=False)
        opt = tf.train.AdamOptimizer(lr, epsilon=1e-3)
        return optimizer.apply_grad_processors(
            opt, [gradproc.ScaleGradient(
                [('aspp.*_conv.*/Wnnnnnn', 10),('aspp.*_conv.*/bnnnnn', 20), ('conv.*/bnnnnn', 2)])]) #TODO



def get_data(name, batch_size):
    isTrain = True if 'train' in name else False
    ds = DataLoader(name = name, max_length=MAX_LENGTH, img_size=IMG_SIZE)

    if isTrain:
        ds = BatchData(ds, batch_size)
        ds = PrefetchDataZMQ(ds, 1)
    else:
        ds = BatchData(ds, 1)
    return ds


def view_data():
    ds = RepeatedData(get_data('train',10), -1)
    ds.reset_state()
    for ims, labels,captions in ds.get_data():
        for im, label,caption in zip(ims, labels,captions):
            cv2.imshow("im", im)
            cv2.imshow("color-label", visualize_label(label,class_num=CLASS_NUM))
            print(caption)
            cv2.waitKey(10000)


def get_config(batch_size):
    logger.auto_set_dir()
    dataset_train = get_data('train', batch_size)
    steps_per_epoch = dataset_train.size()*step_scale

    callbacks = [
        ModelSaver(),
        GPUUtilizationTracker(),
        EstimatedTimeLeft(),
        ScheduledHyperParamSetter('learning_rate', lr_schedule),
        PeriodicTrigger(CalculateMIoU(CLASS_NUM), every_k_epochs=evaluate_every_n_epoch),
        ProgressBar(["cross_entropy_loss", "cost", "wd_cost"]),  # uncomment it to debug for every step
        # RunOp(lambda: tf.add_check_numerics_ops(), run_before=False, run_as_trigger=True, run_step=True)
    ]

    return TrainConfig(
        model=Model(),
        dataflow=dataset_train,
        callbacks=callbacks,
        steps_per_epoch=steps_per_epoch,
        max_epoch=max_epoch,
    )



class CalculateMIoU(Callback):
    def __init__(self, nb_class):
        self.nb_class = nb_class

    def _setup_graph(self):
        self.pred = self.trainer.get_predictor(
            ['image','caption'], ['prob'])

    def _before_train(self):
        pass

    def _trigger(self):
        global args
        self.val_ds = get_data('test',batch_size=args.batch_size)
        self.val_ds.reset_state()

        self.stat = MIoUStatistics(self.nb_class)

        for image, label, caption  in tqdm(self.val_ds.get_data()):
            label = np.squeeze(label)
            image = np.squeeze(image)

            def mypredictor(input_img):
                # input image: 1*H*W*3
                # output : H*W*C
                output = self.pred(input_img[np.newaxis, :, :, :], caption)
                return output[0][0]

            prediction = mypredictor(image)
            prediction = np.argmax(prediction, axis=2)
            self.stat.feed(prediction, label)

        self.trainer.monitors.put_scalar("mIoU", self.stat.mIoU)
        self.trainer.monitors.put_scalar("mean_accuracy", self.stat.mean_accuracy)
        self.trainer.monitors.put_scalar("accuracy", self.stat.accuracy)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='5', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', default="deeplab_resnet_init.ckpt" ,help='load model')
    parser.add_argument('--view', help='view dataset', action='store_true')
    parser.add_argument('--run', help='run model on images')
    parser.add_argument('--batch_size', type=int, default = 1, help='batch_size')
    parser.add_argument('--output', help='fused output filename. default to out-fused.png')
    args = parser.parse_args()
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


    if args.view:
        view_data()
    else:
        config = get_config(args.batch_size)
        if args.load:
            config.session_init = get_model_loader(args.load)
        launch_train_with_config(
            config,
            SyncMultiGPUTrainer(max(get_nr_gpu(), 1)))
