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
from tensorpack.utils.segmentation.segmentation import predict_slider, visualize_label
from tensorpack.utils.stats import MIoUStatistics
from tensorpack.utils import logger
from tensorpack.tfutils import optimizer
from tensorpack.tfutils.summary import add_moving_summary, add_param_summary
import tensorpack.tfutils.symbolic_functions as symbf
from tqdm import tqdm
from LSTM_model_convlstm_p543 import LSTM_model


CLASS_NUM = 21
IMG_SIZE = 320
IGNORE_LABEL = 255
VOCAB_SIZE = 999
STEP_NUM = 49 # equal Max Length

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

    def _get_inputs(self):
        ## Set static shape so that tensorflow knows shape at compile time.
        return [InputDesc(tf.float32, [None, IMG_SIZE, IMG_SIZE, 3], 'image'),
                InputDesc(tf.int32, [None, IMG_SIZE, IMG_SIZE], 'gt'),
                InputDesc(tf.int32, [None, STEP_NUM], 'caption'),]

    def _build_graph(self, inputs):
        image, label, caption = inputs
        image = image - tf.constant([104, 116, 122], dtype='float32')
        mode = "train" if get_current_tower_context().is_training else "val"

        model = LSTM_model(image, caption, mode=mode, vocab_size=VOCAB_SIZE, weights="deeplab")
        predict = model.up

        label = tf.identity(label, name="label")


        costs = []
        prob = tf.identity(predict, name='prob')
        label4d = tf.expand_dims(label, 3, name='label4d')

        cost = symbf.softmax_cross_entropy_with_ignore_label(logits=prob, label=label4d,
                                                             class_num=CLASS_NUM)
        prediction = tf.argmax(prob, axis=-1,name="prediction")
        cost = tf.reduce_mean(cost, name='cross_entropy_loss')  # the average cross-entropy loss
        costs.append(cost)

        if get_current_tower_context().is_training:
            wd_w = tf.train.exponential_decay(2e-4, get_global_step_var(),
                                              80000, 0.7, True)
            wd_cost = tf.multiply(wd_w, regularize_cost('.*/W', tf.nn.l2_loss), name='wd_cost') #TODO
            costs.append(wd_cost)

            self.cost = tf.add_n(costs, name='cost')
            add_moving_summary(costs + [self.cost])

    def _get_optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=2.5e-4, trainable=False)
        opt = tf.train.AdamOptimizer(lr, epsilon=1e-3)
        return optimizer.apply_grad_processors(
            opt, [gradproc.ScaleGradient(
                [('aspp.*_conv.*/W', 10),('aspp.*_conv.*/b', 20), ('conv.*/b', 2)])]) #TODO


def get_data(name, data_dir, meta_dir, batch_size):
    isTrain = name == 'train'
    ds = dataset.PascalVOC12(data_dir, meta_dir, name, shuffle=True)

    class RandomCropWithPadding(imgaug.ImageAugmentor):
        def _get_augment_params(self, img):
            self.h0 = img.shape[0]
            self.w0 = img.shape[1]

            if IMG_SIZE > self.h0:
                top = (IMG_SIZE - self.h0) / 2
                bottom = (IMG_SIZE - self.h0) - top
            else:
                top = 0
                bottom = 0

            if IMG_SIZE > self.w0:
                left = (IMG_SIZE - self.w0) / 2
                right = (IMG_SIZE - self.w0) - left
            else:
                left = 0
                right = 0
            new_shape = (top + bottom + self.h0, left + right + self.w0)
            diffh = new_shape[0] - IMG_SIZE
            assert diffh >= 0
            crop_start_h = 0 if diffh == 0 else self.rng.randint(diffh)
            diffw = new_shape[1] - IMG_SIZE
            assert diffw >= 0
            crop_start_w = 0 if diffw == 0 else self.rng.randint(diffw)
            return (top, bottom, left, right, crop_start_h, crop_start_w)

        def _augment(self, img, param):
            top, bottom, left, right, crop_start_h, crop_start_w = param
            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=IGNORE_LABEL)
            assert crop_start_h + IMG_SIZE <= img.shape[0], crop_start_w + IMG_SIZE <= img.shape[1]
            return img[crop_start_h:crop_start_h + IMG_SIZE, crop_start_w:crop_start_w + IMG_SIZE]


    if isTrain:
        shape_aug = [
            RandomCropWithPadding(),
            imgaug.Flip(horiz=True),
        ]
    else:
        shape_aug = []
        pass
    ds = AugmentImageComponents(ds, shape_aug, (0, 1), copy=False)


    if isTrain:
        ds = BatchData(ds, batch_size)
        ds = PrefetchDataZMQ(ds, 1)
    else:
        ds = BatchData(ds, 1)
    return ds


def view_data(data_dir, meta_dir, batch_size):
    ds = RepeatedData(get_data('train',data_dir, meta_dir, batch_size), -1)
    ds.reset_state()
    for ims, labels in ds.get_data():
        for im, label in zip(ims, labels):
            #aa = visualize_label(label)
            #pass
            cv2.imshow("im", im / 255.0)
            cv2.imshow("raw-label", label)
            cv2.imshow("color-label", visualize_label(label))
            cv2.waitKey(0)


def get_config(data_dir, meta_dir, batch_size):
    logger.auto_set_dir()
    dataset_train = get_data('train', data_dir, meta_dir, batch_size)
    steps_per_epoch = dataset_train.size() * 9
    dataset_val = get_data('val', data_dir, meta_dir, batch_size)

    return TrainConfig(
        dataflow=dataset_train,
        callbacks=[
            ModelSaver(),
            ScheduledHyperParamSetter('learning_rate', [(2, 1.25e-4), (4, 5e-5), (6, 2.5e-5)]),
            HumanHyperParamSetter('learning_rate'),
            PeriodicTrigger(CalculateMIoU(CLASS_NUM), every_k_epochs=1),
            ProgressBar(["cross_entropy_loss","cost","wd_cost"])#uncomment it to debug for every step
        ],
        model=Model(),
        steps_per_epoch=steps_per_epoch,
        max_epoch=10,
    )


def run(model_path, image_path, output):
    return #TODO

class CalculateMIoU(Callback):
    def __init__(self, nb_class):
        self.nb_class = nb_class

    def _setup_graph(self):
        self.pred = self.trainer.get_predictor(
            ['image'], ['prob'])

    def _before_train(self):
        pass

    def _trigger(self):
        global args
        self.val_ds = get_data('val', args.data_dir, args.meta_dir, args.batch_size)
        self.val_ds.reset_state()

        self.stat = MIoUStatistics(self.nb_class)

        for image, label in tqdm(self.val_ds.get_data()):
            label = np.squeeze(label)
            image = np.squeeze(image)
            prediction = predict_slider(image, self.pred, self.nb_class, tile_size=IMG_SIZE)
            prediction = np.argmax(prediction, axis=2)
            self.stat.feed(prediction, label)

        self.trainer.monitors.put_scalar("mIoU", self.stat.mIoU)
        self.trainer.monitors.put_scalar("mean_accuracy", self.stat.mean_accuracy)
        self.trainer.monitors.put_scalar("accuracy", self.stat.accuracy)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--data_dir', default="/data1/dataset/pascalvoc2012/VOC2012trainval/VOCdevkit/VOC2012",
                        help='dataset dir')
    parser.add_argument('--meta_dir', default="pascalvoc12", help='meta dir')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--view', help='view dataset', action='store_true')
    parser.add_argument('--run', help='run model on images')
    parser.add_argument('--batch_size', type=int, default = 16, help='batch_size')
    parser.add_argument('--output', help='fused output filename. default to out-fused.png')
    args = parser.parse_args()
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


    if args.view:
        view_data(args.data_dir,args.meta_dir,args.batch_size)
    elif args.run:
        run(args.load, args.run, args.output)
    else:
        config = get_config(args.data_dir,args.meta_dir,args.batch_size)
        if args.load:
            config.session_init = get_model_loader(args.load)
        launch_train_with_config(
            config,
            SyncMultiGPUTrainer(max(get_nr_gpu(), 1)))
