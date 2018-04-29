#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: deeplabv2.py
# Author: Tao Hu <taohu620@gmail.com>

import cv2
import tensorflow as tf
import argparse
from six.moves import zip
import os
import numpy as np

os.environ['TENSORPACK_TRAIN_API'] = 'v2'   # will become default soon
from tensorpack import *
from tensorpack.dataflow.dataset import Camvid
from tensorpack.utils.gpu import get_nr_gpu
from tensorpack.utils.segmentation.segmentation import predict_slider, visualize_label, predict_scaler
from tensorpack.utils.stats import MIoUStatistics
from tensorpack.utils import logger
from tensorpack.tfutils import optimizer
from tensorpack.tfutils.summary import add_moving_summary, add_param_summary
from densenet_v1_deepsupervision import densenet
slim = tf.contrib.slim

from tensorpack.gist import memory_saving_gradients
from tensorflow.python.ops import gradients
#https://github.com/openai/gradient-checkpointing/issues/4
# monkey patch tf.gradients to point to our custom version, with automatic checkpoint selection
def gradients_memory(ys, xs, grad_ys=None, **kwargs):
    logger.warn("you are using sub-linear gradient~~~")
    #https://github.com/openai/gradient-checkpointing/issues/5
    return memory_saving_gradients.gradients(ys, xs, grad_ys, checkpoints='speed', **kwargs)
gradients.__dict__["gradients"] = gradients_memory

from tqdm import tqdm
from seg_utils import RandomCropWithPadding, softmax_cross_entropy_with_ignore_label



CLASS_NUM = Camvid.class_num()
CROP_SIZE = 321
batch_size = 40

IGNORE_LABEL = 255

GROWTH_RATE = 48
first_batch_lr = 1e-3
lr_schedule = [(4, 1e-4), (8, 1e-5)]
epoch_scale = 800 #640
max_epoch = 10
lr_multi_schedule = [('nothing', 5),('nothing',10)]
evaluate_every_n_epoch = 1

def get_data(name, data_dir, meta_dir, batch_size):

    isTrain = True if 'train' in name else False

    ds = Camvid(data_dir, meta_dir, name, shuffle=True)

    if isTrain:#special augmentation
        shape_aug = [imgaug.RandomResize(xrange=(0.7, 1.5), yrange=(0.7, 1.5),
                            aspect_ratio_thres=0.15),
                     RandomCropWithPadding(CROP_SIZE,IGNORE_LABEL),
                     imgaug.Flip(horiz=True),
                     ]
    else:
        shape_aug = []

    ds = AugmentImageComponents(ds, shape_aug, (0, 1), copy=False)


    #ds = FakeData([[CROP_SIZE, CROP_SIZE, 3], [CROP_SIZE, CROP_SIZE]], 5000, random=False, dtype='uint8')
    if isTrain:
        ds = BatchData(ds, batch_size)
        ds = PrefetchDataZMQ(ds, 1)
    else:
        ds = BatchData(ds, 1)
    return ds

class Model(ModelDesc):

    def _get_inputs(self):
        ## Set static shape so that tensorflow knows shape at compile time.
        return [InputDesc(tf.float32, [None, CROP_SIZE, CROP_SIZE, 3], 'image'),
                InputDesc(tf.int32, [None, CROP_SIZE, CROP_SIZE], 'gt')]

    def _build_graph(self, inputs):
        def mydensenet(image):
            # Prepare parameters for DenseNet
            # assert (args.num_layers - 4) % 3 == 0, 'The number of layers is wrong'
            # num_units = (args.num_layers - 4) // 3
            # blocks = [num_units, num_units, num_units]
            blocks = [6, 8, 8, 8]
            # blocks = [6, 12, 48, 32]
            # blocks = [6, 12, 64, 48]
            rate = [1, 1, 2, 4]
            stride = [2, 2, 1, 1]

            ctx = get_current_tower_context()
            logger.info("current ctx.is_training: {}".format(ctx.is_training))

            # Training
            net, end_points = densenet(image,
                                       rate=rate,
                                       stride=stride,
                                       blocks=blocks,
                                       growth=args.growth_rate,
                                       drop=0.2,
                                       weight_decay=0.00001,
                                       num_classes=CLASS_NUM,
                                       compress = 1,
                                       stem = 1,
                                       denseindense=6,
                                       remove_latter_pooling=True,
                                       data_name='imagenet',
                                       is_training=ctx.is_training,
                                       scope='densenet_L{}_k{}'.format(args.num_layers,
                                                                       args.growth_rate))

            return net

        image, label = inputs
        image = image - tf.constant([104, 116, 122], dtype='float32')
        label = tf.identity(label, name="label")

        predict_list = mydensenet(image)
        for ii, p in enumerate(predict_list):
            predict_list[ii] = tf.image.resize_bilinear(predict_list[ii], image.shape[1:3])

        predict = predict_list[-1]
        costs = []
        prob = tf.nn.softmax(predict, name='prob')

        label4d = tf.expand_dims(label, 3, name='label4d')
        new_size = prob.get_shape()[1:3]
        # label_resized = tf.image.resize_nearest_neighbor(label4d, new_size)

        cost = 0
        for jj,p in enumerate(predict_list):
            current_predict = predict_list[jj]
            cost += softmax_cross_entropy_with_ignore_label(logits=current_predict, label=label4d,
                                                       class_num=CLASS_NUM)
        cost = cost/len(predict_list)

        prediction = tf.argmax(prob, axis=-1, name="prediction")
        cost = tf.reduce_mean(cost, name='cross_entropy_loss')  # the average cross-entropy loss
        costs.append(cost)

        if get_current_tower_context().is_training:
            # wd_w = tf.train.exponential_decay(wd, get_global_step_var(),
            #                                  80000, 0.7, True)
            # wd_cost = tf.multiply(wd_w, regularize_cost('.*/W', tf.nn.l2_loss), name='wd_cost')
            wd_cost = tf.add_n(slim.losses.get_regularization_losses(), name='wd_cost')
            costs.append(wd_cost)

            self.cost = tf.add_n(costs, name='cost')


    def _get_optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=first_batch_lr, trainable=False)
        opt = tf.train.AdamOptimizer(lr, epsilon=2.5e-4)
        return optimizer.apply_grad_processors(
            opt, [gradproc.ScaleGradient(
                lr_multi_schedule)])




def view_data(data_dir, meta_dir, batch_size):
    ds = RepeatedData(get_data('train',data_dir, meta_dir, batch_size), -1)
    ds.reset_state()
    for ims, labels in ds.get_data():
        for im, label in zip(ims, labels):
            #aa = visualize_label(label)
            #pass
            cv2.imshow("im", im / 255.0)
            cv2.imshow("raw-label", label)
            cv2.imshow("color-label", visualize_label(label,ignore_label=IGNORE_LABEL))
            cv2.waitKey(3000)


def get_config(data_dir, meta_dir, batch_size):
    logger.auto_set_dir()
    nr_tower = max(get_nr_gpu(), 1)
    dataset_train = get_data('train_val', data_dir, meta_dir, batch_size)
    steps_per_epoch = dataset_train.size() * epoch_scale

    return TrainConfig(
        dataflow=dataset_train,
        callbacks=[
            ModelSaver(),
            ScheduledHyperParamSetter('learning_rate', lr_schedule),
            HumanHyperParamSetter('learning_rate'),
            PeriodicTrigger(CalculateMIoU(CLASS_NUM), every_k_epochs=evaluate_every_n_epoch),
            ProgressBar(["cross_entropy_loss","cost","wd_cost"])#uncomment it to debug for every step
        ],
        model=Model(),
        steps_per_epoch=steps_per_epoch,
        max_epoch=max_epoch,
        nr_tower = nr_tower
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

def proceed_validation(args, is_save = False, is_densecrf = False):
    import cv2
    ds = Camvid(args.data_dir, args.meta_dir, "test")
    ds = BatchData(ds, 1)

    pred_config = PredictConfig(
        model=Model(),
        session_init=get_model_loader(args.load),
        input_names=['image'],
        output_names=['prob'])
    predictor = OfflinePredictor(pred_config)

    i = 0
    stat = MIoUStatistics(CLASS_NUM)
    logger.info("start validation....")

    def mypredictor(input_img):
        # input image: 1*H*W*3
        # output : H*W*C
        output = predictor(input_img)
        return output[0][0]

    for image, label in tqdm(ds.get_data()):
        label = np.squeeze(label)
        image = np.squeeze(image)
        prediction = predict_scaler(image, mypredictor, scales=[0.5,0.75, 1, 1.25, 1.5], classes=CLASS_NUM, tile_size=CROP_SIZE, is_densecrf = is_densecrf)
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
            ['image'], ['prob'])

    def _before_train(self):
        pass

    def _trigger(self):
        global args
        self.val_ds = get_data('test', args.data_dir, args.meta_dir, args.batch_size)
        self.val_ds.reset_state()

        self.stat = MIoUStatistics(self.nb_class)

        def mypredictor(input_img):
            # input image: 1*H*W*3
            # output : H*W*C
            output = self.pred(input_img[np.newaxis, :, :, :])
            return output[0][0]

        for image, label in tqdm(self.val_ds.get_data()):
            label = np.squeeze(label)
            image = np.squeeze(image)
            prediction = predict_scaler(image, mypredictor, scales=[0.5,0.75, 1, 1.25, 1.5], classes=CLASS_NUM, tile_size=CROP_SIZE,
                           is_densecrf=False)
            prediction = np.argmax(prediction, axis=2)
            self.stat.feed(prediction, label)

        self.trainer.monitors.put_scalar("mIoU", self.stat.mIoU)
        self.trainer.monitors.put_scalar("mean_accuracy", self.stat.mean_accuracy)
        self.trainer.monitors.put_scalar("accuracy", self.stat.accuracy)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default="0", help='comma separated list of GPU(s) to use.')
    parser.add_argument('--data_dir', default="/data1/dataset/SegNet-Tutorial",
                        help='dataset dir')
    parser.add_argument('--meta_dir', default="metadata/camvid", help='meta dir')
    #parser.add_argument('--load', default="../resnet101.npz", help='load model')
    parser.add_argument('--growth_rate', default= GROWTH_RATE, help='growth_rate')
    parser.add_argument('--num_layers', default=121, help='num_layers')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--view', help='view dataset', action='store_true')
    parser.add_argument('--run', help='run model on images')
    parser.add_argument('--batch_size', type=int, default = batch_size, help='batch_size')
    parser.add_argument('--output', help='fused output filename. default to out-fused.png')
    parser.add_argument('--validation', action='store_true', help='validate model on validation images')
    args = parser.parse_args()
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


    if args.view:
        view_data(args.data_dir,args.meta_dir,args.batch_size)
    elif args.run:
        run(args.load, args.run, args.output)
    elif args.validation:
        proceed_validation(args)
    else:
        config = get_config(args.data_dir,args.meta_dir,args.batch_size)
        if args.load:
            config.session_init = get_model_loader(args.load)
        launch_train_with_config(
            config,
            SyncMultiGPUTrainer(max(get_nr_gpu(), 1)))
