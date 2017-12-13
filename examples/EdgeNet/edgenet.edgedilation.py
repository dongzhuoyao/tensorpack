#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: hed.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import cv2
import tensorflow as tf
import argparse
from six.moves import zip
import os

os.environ['TENSORPACK_TRAIN_API'] = 'v2'   # will become default soon
from tensorpack import *
from tensorpack.dataflow import dataset
from tensorpack.utils.gpu import get_nr_gpu
from tensorpack.tfutils import optimizer
from tensorpack.tfutils.summary import add_moving_summary, add_param_summary
from utils import *

crop_size = 256
BATCH_SIZE = 10

def class_balanced_sigmoid_cross_entropy(logits, label, name='cross_entropy_loss'):
    """
    The class-balanced cross entropy loss,
    as in `Holistically-Nested Edge Detection
    <http://arxiv.org/abs/1504.06375>`_.

    Args:
        logits: of shape (b, ...).
        label: of the same shape. the ground truth in {0,1}.
    Returns:
        class-balanced cross entropy loss.
    """
    with tf.name_scope('class_balanced_sigmoid_cross_entropy'):
        y = tf.cast(label, tf.float32)

        count_neg = tf.reduce_sum(1. - y)
        count_pos = tf.reduce_sum(y)
        beta = count_neg / (count_neg + count_pos)

        pos_weight = beta / (1 - beta)
        cost = tf.nn.weighted_cross_entropy_with_logits(logits=logits, targets=y, pos_weight=pos_weight)
        cost = tf.reduce_mean(cost * (1 - beta))
        zero = tf.equal(count_pos, 0.0)
    return tf.where(zero, 0.0, cost, name=name)


class Model(ModelDesc):
    def _get_inputs(self):
        return [InputDesc(tf.float32, [None, None, None, 3], 'image'),
                InputDesc(tf.int32, [None, None, None], 'edgemap')]

    def _build_graph(self, inputs):
        image, edgemap = inputs
        image = image - tf.constant([104, 116, 122], dtype='float32')
        edgemap = tf.expand_dims(edgemap, 3, name='edgemap4d')

        def branch(name, l, up):
            with tf.variable_scope(name):
                l = Conv2D('convfc', l, 1, kernel_shape=1, nl=tf.identity,
                           use_bias=True,
                           W_init=tf.constant_initializer(),
                           b_init=tf.constant_initializer())
                while up != 1:
                    l = BilinearUpSample('upsample{}'.format(up), l, 2)
                    up = up / 2
                return l

        with argscope(Conv2D, kernel_shape=3, nl=tf.nn.relu):
            l = Conv2D('conv1_1', image, 64)
            l = Conv2D('conv1_2', l, 64)
            b1 = branch('branch1', l, 1)
            l = MaxPooling('pool1', l, 2)



            l = Conv2D('conv2_1', l, 128)
            l = Conv2D('conv2_2', l, 128)
            b2 = branch('branch2', l, 2)
            l = MaxPooling('pool2', l, 2)

            l = Conv2D('conv3_1', l, 256)
            l = Conv2D('conv3_2', l, 256)
            l = Conv2D('conv3_3', l, 256)
            b3 = branch('branch3', l, 4)
            l = MaxPooling('pool3', l, 2)

            l = Conv2D('conv4_1', l, 512)
            l = Conv2D('conv4_2', l, 512)
            l = Conv2D('conv4_3', l, 512)
            b4 = branch('branch4', l, 8)
            l = MaxPooling('pool4', l, 2)

            l = Conv2D('conv5_1', l, 512)
            l = Conv2D('conv5_2', l, 512)
            l = Conv2D('conv5_3', l, 512)
            b5 = branch('branch5', l, 16)

        final_map = Conv2D('convfcweight',
                           tf.concat([b1, b2, b3, b4, b5], 3), 1, 1,
                           W_init=tf.constant_initializer(0.2),
                           use_bias=False, nl=tf.identity)
        costs = []
        for idx, b in enumerate([b1, b2, b3, b4, b5, final_map]):
            output = tf.nn.sigmoid(b, name='output{}'.format(idx + 1))
            xentropy = class_balanced_sigmoid_cross_entropy(
                b, edgemap,
                name='xentropy{}'.format(idx + 1))
            costs.append(xentropy)

        # some magic threshold
        pred = tf.cast(tf.greater(output, 0.5), tf.int32, name='prediction')
        wrong = tf.cast(tf.not_equal(pred, edgemap), tf.float32)
        wrong = tf.reduce_mean(wrong, name='train_error')

        if get_current_tower_context().is_training:
            wd_w = tf.train.exponential_decay(2e-4, get_global_step_var(),
                                              80000, 0.7, True)
            wd_cost = tf.multiply(wd_w, regularize_cost('.*/W', tf.nn.l2_loss), name='wd_cost')
            costs.append(wd_cost)

            add_param_summary(('.*/W', ['histogram']))   # monitor W
            self.cost = tf.add_n(costs, name='cost')
            add_moving_summary(costs + [wrong, self.cost])

    def _get_optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=3e-5, trainable=False)
        opt = tf.train.AdamOptimizer(lr, epsilon=1e-3)
        return optimizer.apply_grad_processors(
            opt, [gradproc.ScaleGradient(
                [('convfcweight.*', 0.1), ('conv5_.*', 5)])])


def get_data(data_dir,meta_dir,name,batch_size=-1,crop_size=-1):
    isTrain = name == 'train'
    ds = dataset.Ningbo(data_dir, meta_dir, name, shuffle=True)

    if isTrain:
        shape_aug = [
            #imgaug.RandomResize(xrange=(0.7, 1.5), yrange=(0.7, 1.5),
            #                    aspect_ratio_thres=0.15),
            #imgaug.RandomCropWithPadding((crop_size, crop_size))
        ]
    else:
        # the original image shape (321x481) in BSDS is not a multiple of 16
        shape_aug = []
    ds = AugmentImageComponents(ds, shape_aug, (0, 1), copy=False)


    if isTrain:
        augmentors = [
            #imgaug.Brightness(63, clip=False),
            #imgaug.Contrast((0.4, 1.5)),
        ]
        ds = AugmentImageComponent(ds, augmentors, copy=False)
        #ds = BatchDataByShape(ds, 8, idx=0)
        ds = BatchData(ds, BATCH_SIZE, remainder=not isTrain)
        #ds = PrefetchDataZMQ(ds, 1)
    else:
        ds = BatchData(ds, 1)
    return ds


def view_data(data_dir,meta_dir,batch_size,crop_size):
    ds = RepeatedData(get_data(data_dir,meta_dir, 'train',batch_size,crop_size), -1)
    ds.reset_state()
    for ims, labels in ds.get_data():
        for im, label in zip(ims, labels):
            cv2.imshow("image", im / 255.0)
            cv2.imshow("label", visualize_label(label))
            cv2.waitKey()


def get_config(data_dir,meta_dir,batch_size,crop_size, val_crop_size, class_num):
    logger.auto_set_dir()
    dataset_train = get_data(data_dir,meta_dir,'train',batch_size,crop_size)
    steps_per_epoch = dataset_train.size()*10
    dataset_val = get_data(data_dir,meta_dir,'val',batch_size, val_crop_size)

    return TrainConfig(
        dataflow=dataset_train,
        callbacks=[
            ModelSaver(),
            ScheduledHyperParamSetter('learning_rate', [(30, 6e-6), (45, 1e-6), (60, 8e-7)]),
            HumanHyperParamSetter('learning_rate'),
            InferenceRunner(dataset_val,
                            BinaryClassificationStats('prediction', 'edgemap4d')),
            #ProgressBar(["cost", "wd_cost"])  # uncomment it to debug for every step
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


def proceed_validation(args, is_save = False, is_densecrf = False):
    import cv2
    ds = dataset.Ningbo(args.data_dir, args.meta_dir,"val")
    ds = BatchData(ds, 1)

    pred_config = PredictConfig(
        model=Model(),
        session_init=get_model_loader(args.load),
        input_names=['image'],
        output_names=['output' + str(k) for k in range(1, 7)])
    predictor = OfflinePredictor(pred_config)

    from tqdm import tqdm
    i =0
    def to_size(input):
        input[input >= 0.50] = 1
        input[input < 0.50] = 0
        return np.dstack((input*255,input*255,input*255))
    for image, label in tqdm(ds.get_data()):
        i += 1
        #image = image[None, :, :, :].astype('float32')
        outputs = predictor(image)
        label = label[0]
        label = label[:,:,None]
        cv2.imwrite("result/out{}.png".format(i),
                    np.concatenate((image[0],
                                    to_size(label),
                                    to_size(outputs[0][0]),
                                    to_size(outputs[1][0]),
                                    to_size(outputs[2][0]),
                                    to_size(outputs[3][0]),
                                    to_size(outputs[4][0]),
                                    to_size(outputs[5][0])),
                                   axis=1))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default="3", help='comma separated list of GPU(s) to use.')
    parser.add_argument('--data_dir',default="/data_a/dataset/ningbo3539_edge_gt_new", help='dataset dir')
    parser.add_argument('--meta_dir', default="ningbo", help='meta dir')
    #parser.add_argument('--edge_dir', default="/data_a/dataset/ningbo3539_edge_gt_new",  help='edge dir')
    parser.add_argument('--class_num', type=int, default=2)
    parser.add_argument('--batch_size', default=1, type=int, help='batch size')
    parser.add_argument('--crop_size', default=256, type=int, help='crop size')
    parser.add_argument('--val_crop_size', default=512, type=int, help='crop size')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--view', help='view dataset', action='store_true')
    parser.add_argument('--run', help='run model on images')
    parser.add_argument('--validation', action='store_true', help='validate model on validation images')
    parser.add_argument('--output', help='fused output filename. default to out.png')
    args = parser.parse_args()
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

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
        config.nr_tower = max(get_nr_gpu(), 1)
        SyncMultiGPUTrainer(config).train()