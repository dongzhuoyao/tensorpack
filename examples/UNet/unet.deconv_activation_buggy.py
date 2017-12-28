#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: unet.py
# Author: Tao Hu <taohu620@gmail.com>

import cv2
import tensorflow as tf
import argparse
import numpy as np
from six.moves import zip
import os
import sys
from utils import *
from tqdm import tqdm
from tensorpack import *
import tensorpack.tfutils.symbolic_functions as symbf
from tensorpack.dataflow import dataset
from tensorpack.utils.gpu import get_nr_gpu
from tensorpack.tfutils import optimizer
from tensorpack.tfutils.summary import *
from tensorpack.utils.stats import MIoUStatistics
from tensorpack.utils.segmentation import predict_slider, visualize_label, predict_scaler
from tensorpack.utils import logger
from tensorpack.dataflow.imgaug.misc import RandomCropWithPadding

IGNORE_LABEL = 255

class Model(ModelDesc):
    def __init__(self,class_num):
        self.class_num = class_num

    def _get_inputs(self):
        return [InputDesc(tf.float32, [None, None, None, 3], 'image'),
                InputDesc(tf.int32, [None, None, None], 'edgemap')]

    def _build_graph(self, inputs):
        image, edgemap = inputs
        edgemap4d = tf.expand_dims(edgemap, name="edgemap4d", dim=3)
        image = image - tf.constant([104, 116, 122], dtype='float32')


        with argscope(Conv2D, kernel_shape=3, nl=tf.nn.relu):
            conv1 = Conv2D('conv1_1',image,32)
            conv1 = Conv2D('conv1_2', conv1, 32)
            pool1 = MaxPooling('pool1',conv1,2)

            conv2 = Conv2D('conv2_1', pool1, 64)
            conv2 = Conv2D('conv2_2', conv2, 64)
            pool2= MaxPooling('pool2',conv2,2)

            conv3 = Conv2D('conv3_1', pool2, 128)
            conv3 = Conv2D('conv3_2', conv3, 128)
            pool3 = MaxPooling('pool3',conv3,2)

            conv4 = Conv2D('conv4_1', pool3, 256)
            conv4 = Conv2D('conv4_2', conv4, 256)
            pool4 = MaxPooling('pool4',conv4,2)

            conv5 = Conv2D('conv5_1', pool4, 512)
            conv5 = Conv2D('conv5_2', conv5, 512)


            up6 = tf.concat([Deconv2D('deconv6',conv5,256,kernel_shape=2,stride=2),conv4],axis=3)
            conv6 = Conv2D('conv6_1',up6,256)
            conv6 = Conv2D('conv6_2', conv6, 256)

            up7 = tf.concat([Deconv2D('deconv7', conv6, 128,kernel_shape=2,stride=2), conv3], axis=3)
            conv7 = Conv2D('conv7_1', up7, 128)
            conv7 = Conv2D('conv7_2', conv7, 128)

            up8 = tf.concat([Deconv2D('deconv8', conv7, 64,kernel_shape=2,stride=2), conv2], axis=3)
            conv8 = Conv2D('conv8_1', up8, 64)
            conv8 = Conv2D('conv8_2', conv8, 64)

            up9 = tf.concat([Deconv2D('deconv9', conv8, 32,kernel_shape=2,stride=2), conv1], axis=3)
            conv9 = Conv2D('conv9_1', up9, 32)
            conv9 = Conv2D('conv9_2', conv9, 32)

        final_map = Conv2D('convfcweight',
                           conv9,
                           out_channel = self.class_num, kernel_shape = (1,1),
                           W_init=tf.constant_initializer(0.2),
                           use_bias=False, nl=tf.identity)

        final_map = tf.image.resize_bilinear(final_map, tf.shape(image)[1:3, ])
        predict_prob = tf.nn.softmax(final_map,name="predict_prob")
        raw_output_up = tf.argmax(predict_prob, dimension=3)
        pred = tf.expand_dims(raw_output_up, name="predict", dim=3)
        pred = tf.to_int32(pred)
        costs = []

        label = tf.reshape(edgemap4d, [-1])
        pred = tf.reshape(pred,[-1])

        indices = tf.squeeze(tf.where(tf.less_equal(label, self.class_num - 1)), axis=1)
        trimmed_edgemap4d = tf.gather(label, indices)
        trimmed_edgemap4d = tf.expand_dims(trimmed_edgemap4d, dim=1, name="trimmed_edgemap4d")
        trimmed_pred = tf.gather(pred, indices)
        trimmed_pred = tf.expand_dims(trimmed_pred, dim=1, name="trimmed_predict")

        cost = symbf.softmax_cross_entropy_with_ignore_label(logits=final_map, label=edgemap4d, class_num = self.class_num)
        cost = tf.reduce_mean(cost, name='cross_entropy_loss')  # the average cross-entropy loss
        costs.append(cost)


        if get_current_tower_context().is_training:
            wd_w = tf.train.exponential_decay(2e-4, get_global_step_var(),
                                           80000, 0.7, True)
            wd_cost = tf.multiply(wd_w, regularize_cost('.*/W', tf.nn.l2_loss), name='regularization_loss')
            costs.append(wd_cost)

            #add_param_summary(('.*/W', ['histogram']))   # monitor W
            self.cost = tf.add_n(costs, name='total_cost')
            add_moving_summary(costs + [self.cost])

    def _get_optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=2.5e-4, trainable=False)
        opt = tf.train.AdamOptimizer(lr, epsilon=1e-3)
        return optimizer.apply_grad_processors(
            opt, [gradproc.ScaleGradient(
                [('convfcweight.*', 0.1), ('conv5_.*', 5)])])


def get_data(data_dir,meta_dir,name,batch_size=-1,crop_size=-1):
    isTrain = name == 'train'
    ds = dataset.PascalVOC12(data_dir,meta_dir,name, shuffle=True)



    if isTrain:
        shape_aug = [
        imgaug.RandomResize(xrange=(0.7, 1.5), yrange=(0.7, 1.5),
                                aspect_ratio_thres=0.15),
        RandomCropWithPadding(CROP_SIZE, IGNORE_LABEL),
        imgaug.Flip(horiz=True)]
    else:
        shape_aug = []

    ds = AugmentImageComponents(ds, shape_aug, (0, 1), copy=False)

    def f(ds):
        return ds
    if isTrain:
        ds = BatchData(ds, batch_size)
        ds = MapData(ds, f)
        ds = PrefetchDataZMQ(ds, 1)
    else:
        ds = BatchData(ds, 1)
    return ds




def view_data(data_dir, meta_dir, batch_size):
    ds = RepeatedData(get_data(data_dir, meta_dir, 'train', batch_size), -1)
    ds.reset_state()
    for ims, labels in ds.get_data():
        for im, label in zip(ims, labels):
            #aa = visualize_label(label)
            #pass
            cv2.imshow("im", im / 255.0)
            cv2.imshow("raw-label", label)
            cv2.imshow("color-label", visualize_label(label))
            cv2.waitKey(0)

def get_config(data_dir,meta_dir,batch_size,crop_size, val_crop_size, class_num):
    logger.auto_set_dir()
    dataset_train = get_data(data_dir,meta_dir,'train',batch_size,crop_size)
    steps_per_epoch = dataset_train.size()*40
    dataset_val = get_data(data_dir,meta_dir,'val',batch_size, val_crop_size)

    return TrainConfig(
        dataflow=dataset_train,
        callbacks=[
            ModelSaver(max_to_keep = -1),
            ScheduledHyperParamSetter('learning_rate', [(5, 1.25e-4), (8, 6.25e-5)]),
            HumanHyperParamSetter('learning_rate'),
            #InferenceRunner(dataset_val,
            #                [ScalarStats('accuracy')]),
            ProgressBar(["cross_entropy_loss", "regularization_loss", "total_cost"]),  # uncomment it to debug for every step
            PeriodicTrigger(CalculateMIoU(class_num), every_k_epochs=1),
        ],
        model=Model(class_num),
        steps_per_epoch=steps_per_epoch,
        max_epoch=12,
    )



def proceed_validation(args, is_save = True, is_densecrf = False):
    import cv2
    ds = dataset.PascalVOC12(args.data_dir, args.meta_dir, "val")
    ds = BatchData(ds, 1)

    pred_config = PredictConfig(
        model=Model(),
        session_init=get_model_loader(args.load),
        input_names=['image'],
        output_names=['predict_prob'])
    predictor = OfflinePredictor(pred_config)

    i = 0
    stat = MIoUStatistics(CLASS_NUM)
    logger.info("start validation....")
    for image, label in tqdm(ds.get_data()):
        label = np.squeeze(label)
        image = np.squeeze(image)
        prediction = predict_scaler(image, predictor, scales=[0.9, 1, 1.1], classes=CLASS_NUM, tile_size=CROP_SIZE, is_densecrf = is_densecrf)
        prediction = np.argmax(prediction, axis=2)
        stat.feed(prediction, label)

        if is_save:
            cv2.imwrite("result/{}.png".format(i), np.concatenate((image, visualize_label(label), visualize_label(prediction)), axis=1))

        i += 1

    logger.info("mIoU: {}".format(stat.mIoU))
    logger.info("mean_accuracy: {}".format(stat.mean_accuracy))
    logger.info("accuracy: {}".format(stat.accuracy))


class CalculateMIoU(Callback):
    def __init__(self, nb_class):
        self.nb_class = nb_class

    def _setup_graph(self):
        self.pred = self.trainer.get_predictor(
            ['image'], ['predict_prob'])

    def _before_train(self):
        pass

    def _trigger(self):
        global args
        self.val_ds = get_data( args.data_dir, args.meta_dir, 'val', args.batch_size)
        self.val_ds.reset_state()

        self.stat = MIoUStatistics(self.nb_class)

        for image, label in tqdm(self.val_ds.get_data()):
            label = np.squeeze(label)
            image = np.squeeze(image)
            prediction = predict_slider(image, self.pred, self.nb_class, tile_size=CROP_SIZE)
            prediction = np.argmax(prediction, axis=2)
            self.stat.feed(prediction, label)

        self.trainer.monitors.put_scalar("mIoU", self.stat.mIoU)
        self.trainer.monitors.put_scalar("mean_accuracy", self.stat.mean_accuracy)
        self.trainer.monitors.put_scalar("accuracy", self.stat.accuracy)


def run(model_path, image_path, output, val_crop_size, class_num):
    pred_config = PredictConfig(
        model=Model(class_num),
        session_init=get_model_loader(model_path),
        input_names=['image'],
        output_names=['predict_prob'])
    predictor = OfflinePredictor(pred_config)
    im = cv2.imread(image_path)
    assert im is not None
    outputs = predict_sliding(im, predictor, class_num, (val_crop_size,val_crop_size))
    outputs = np.argmax(outputs, axis=2)[:, :, None]
    pred = decode_image_label(outputs)
    if output is None:
            cv2.imwrite("predict.png", pred)
    else:
            cv2.imwrite(output, pred)

def proceed_validation(args, is_save = True, is_densecrf = False):
    import cv2
    ds = dataset.PascalVOC12(args.data_dir, args.meta_dir, "val")
    ds = BatchData(ds, 1)

    pred_config = PredictConfig(
        model=Model(args.class_num),
        session_init=get_model_loader(args.load),
        input_names=['image'],
        output_names=['predict_prob'])
    predictor = OfflinePredictor(pred_config)

    i = 0
    stat = MIoUStatistics(args.class_num)
    logger.info("start validation....")
    for image, label in tqdm(ds.get_data()):
        label = np.squeeze(label)
        image = np.squeeze(image)
        prediction = predict_scaler(image, predictor, scales=[1], classes=args.class_num, tile_size=CROP_SIZE, is_densecrf = is_densecrf)
        prediction = np.argmax(prediction, axis=2).astype('uint8')
        stat.feed(prediction, label)

        if is_save:
            edge = cv2.Canny(visualize_label(prediction), 100, 200)
            edge = np.stack((edge, edge, edge), axis=2)
            cv2.imwrite("result/{}.png".format(i), np.concatenate((image, visualize_label(label), visualize_label(prediction), edge), axis=1))

        i += 1

    logger.info("mIoU: {}".format(stat.mIoU))
    logger.info("mean_accuracy: {}".format(stat.mean_accuracy))
    logger.info("accuracy: {}".format(stat.accuracy))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default="3", help='comma separated list of GPU(s) to use.')
    parser.add_argument('--data_dir', default="/data_a/dataset/ningbo3539/", help='dataset dir')
    parser.add_argument('--meta_dir', default="ningbo", help='meta dir')
    parser.add_argument('--class_num', type=int, default=2)
    parser.add_argument('--batch_size', default=24, type=int, help='batch size')
    parser.add_argument('--crop_size', default=256, type=int, help='crop size')
    parser.add_argument('--val_crop_size', default=256, type=int, help='crop size')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--view', help='view dataset', action='store_true')
    parser.add_argument('--run', help='run model on images')
    parser.add_argument('--validation', action='store_true', help='validate model on validation images')
    parser.add_argument('--output', help='fused output filename. default to out.png')
    args = parser.parse_args()
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    CROP_SIZE = args.crop_size

    if args.view:
        view_data(args.data_dir,args.meta_dir,args.batch_size,args.crop_size)
    elif args.run:
        run(args.load, args.run, args.output, args.val_crop_size, args.class_num)
    elif args.validation:
        proceed_validation(args)
    else:
        config = get_config(args.data_dir,args.meta_dir,args.batch_size,args.crop_size,args.val_crop_size,args.class_num)
        if args.load:
            config.session_init = get_model_loader(args.load)
        launch_train_with_config(
            config,
            SyncMultiGPUTrainer(max(get_nr_gpu(), 1)))
