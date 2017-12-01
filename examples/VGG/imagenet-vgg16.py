#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: imagenet-resnet.py

import argparse
import os
import tensorflow as tf

os.environ['TENSORPACK_TRAIN_API'] = 'v2'   # will become default soon
from tensorpack import logger, QueueInput
from tensorpack.models import *
from tensorpack.callbacks import *
from tensorpack.train import (
    TrainConfig, SyncMultiGPUTrainerParameterServer, launch_train_with_config)
from tensorpack.dataflow import FakeData
from tensorpack.tfutils import argscope, get_model_loader
from tensorpack.utils.gpu import get_nr_gpu

from imagenet_utils import (
    fbresnet_augmentor, get_imagenet_dataflow, ImageNetModel,
    eval_on_ILSVRC12)

TOTAL_BATCH_SIZE = 128


class Model(ImageNetModel):
    def __init__(self, data_format='NHWC'):
        super(Model, self).__init__(data_format)

    def get_logits(self, image):
        with argscope(Conv2D, kernel_shape=3, nl=tf.nn.relu):
            l = Conv2D('conv1_1', image, 64)
            l = Conv2D('conv1_2', l, 64)
            l = MaxPooling('pool1', l, 2)

            l = Conv2D('conv2_1', l, 128)
            l = Conv2D('conv2_2', l, 128)
            l = MaxPooling('pool2', l, 2)

            l = Conv2D('conv3_1', l, 256)
            l = Conv2D('conv3_2', l, 256)
            l = Conv2D('conv3_3', l, 256)
            l = MaxPooling('pool3', l, 2)

            l = Conv2D('conv4_1', l, 512)
            l = Conv2D('conv4_2', l, 512)
            l = Conv2D('conv4_3', l, 512)
            l = MaxPooling('pool4', l, 2)

            l = Conv2D('conv5_1', l, 512)
            l = Conv2D('conv5_2', l, 512)
            l = Conv2D('conv5_3', l, 512)
            l = MaxPooling('pool5', l, 2)

            l = FullyConnected('linear', l, 4096, nl=tf.nn.relu)
            l = Dropout("dropout1",l , 0.5)

            l = FullyConnected('linear2', l, 4096, nl=tf.nn.relu)
            l = Dropout("dropout2", l, 0.5)

            l = FullyConnected('linear3', l, 1000, nl=tf.identity)

            return l



def get_data(name, batch):
    isTrain = name == 'train'
    augmentors = fbresnet_augmentor(isTrain)
    return get_imagenet_dataflow(
        args.data, name, batch, augmentors)


def get_config(model, fake=False):
    nr_tower = max(get_nr_gpu(), 1)
    batch = TOTAL_BATCH_SIZE // nr_tower

    if fake:
        logger.info("For benchmark, batch size is fixed to 64 per tower.")
        dataset_train = FakeData(
            [[64, 224, 224, 3], [64]], 1000, random=False, dtype='uint8')
        callbacks = []
    else:
        logger.info("Running on {} towers. Batch size per tower: {}".format(nr_tower, batch))
        dataset_train = get_data('train', batch)
        dataset_val = get_data('val', batch)
        callbacks = [
            ModelSaver(),
            ScheduledHyperParamSetter('learning_rate',
                                      [(30, 1e-2), (60, 1e-3), (85, 1e-4), (95, 1e-5), (105, 1e-6)]),
            HumanHyperParamSetter('learning_rate'),
            ProgressBar(["l2_regularize_loss", "xentropy-loss", "wd_cost"])  # uncomment it to debug for every step
        ]
        infs = [ClassificationError('wrong-top1', 'val-error-top1'),
                ClassificationError('wrong-top5', 'val-error-top5')]
        if nr_tower == 1:
            # single-GPU inference with queue prefetch
            callbacks.append(InferenceRunner(QueueInput(dataset_val), infs))
        else:
            # multi-GPU inference (with mandatory queue prefetch)
            callbacks.append(DataParallelInferenceRunner(
                dataset_val, infs, list(range(nr_tower))))

    return TrainConfig(
        model=model,
        dataflow=dataset_train,
        callbacks=callbacks,
        steps_per_epoch=5000,
        max_epoch=110,
        nr_tower=nr_tower
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--data', default="/data_a/dataset/imagenet2012", help='ILSVRC dataset dir')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--fake', help='use fakedata to test or benchmark this model', action='store_true')
    parser.add_argument('--data_format', help='specify NCHW or NHWC',
                        type=str, default='NHWC')
    parser.add_argument('--eval', action='store_true')
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


    model = Model(args.data_format)
    if args.eval:
        batch = 128    # something that can run on one gpu
        ds = get_data('val', batch)
        eval_on_ILSVRC12(model, get_model_loader(args.load), ds)
    else:
        logger.set_logger_dir(
            os.path.join('train_log', 'imagenet-vgg'))

        config = get_config(model, fake=args.fake)
        if args.load:
            config.session_init = get_model_loader(args.load)
        trainer = SyncMultiGPUTrainerParameterServer(max(get_nr_gpu(), 1))
        launch_train_with_config(config, trainer)
