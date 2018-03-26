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
from tensorpack.utils import logger
import OneShotDatasetTwoBranch
from deeplabv2_dilation6_new import deeplabv2
import tensorflow as tf
slim = tf.contrib.slim
from sess_utils import my_get_model_loader

max_epoch = 6
weight_decay = 5e-4
batch_size = 12
LR = 1e-4
CLASS_NUM = 2
evaluate_every_n_epoch = 1
support_image_size =(321, 321)
query_image_size = (321, 321)

def get_data(name,batch_size=1):
    isTrain = True if 'train' in name else False
    ds = OneShotDatasetTwoBranch.OneShotDatasetTwoBranch(name)

    def data_prepare(ds):
        if isTrain:
            first_image = cv2.imread(ds[0][0], cv2.IMREAD_COLOR)
            first_label = cv2.imread(ds[1][0], cv2.IMREAD_GRAYSCALE)
            second_image = cv2.imread(ds[2], cv2.IMREAD_COLOR)
            second_label = cv2.imread(ds[3], cv2.IMREAD_GRAYSCALE)
            metadata = ds[4]
            class_id = metadata['class_id']
            first_label = np.equal(first_label,class_id).astype(np.uint8)
            second_label = np.equal(second_label,class_id).astype(np.uint8)


            first_image = cv2.resize(first_image,support_image_size)
            first_label = cv2.resize(first_label, support_image_size,interpolation=cv2.INTER_NEAREST)
            second_image = cv2.resize(second_image, support_image_size)
            second_label = cv2.resize(second_label, support_image_size,interpolation=cv2.INTER_NEAREST)

            first_image_masked = first_image*first_label[:,:,np.newaxis]

            return first_image_masked,second_image,second_label
        else:
            k_shots = len(ds[0])
            metadata = ds[4]
            class_id = metadata['class_id']
            first_image_masks = []
            for kk in range(k_shots):
                first_image = cv2.imread(ds[0][kk], cv2.IMREAD_COLOR)
                first_label = cv2.imread(ds[1][kk], cv2.IMREAD_GRAYSCALE)
                first_label = np.equal(first_label, class_id).astype(np.uint8)
                first_image = cv2.resize(first_image, support_image_size)
                first_label = cv2.resize(first_label, support_image_size, interpolation=cv2.INTER_NEAREST)
                first_image_masked = first_image * first_label[:, :, np.newaxis]
                first_image_masks.append(first_image_masked)

            second_image = cv2.imread(ds[2], cv2.IMREAD_COLOR)
            second_label = cv2.imread(ds[3], cv2.IMREAD_GRAYSCALE)
            second_label = np.equal(second_label, class_id).astype(np.uint8)
            #second_image = cv2.resize(second_image, support_image_size)
            #second_label = cv2.resize(second_label, support_image_size, interpolation=cv2.INTER_NEAREST)
            return first_image_masks, second_image, second_label


    if isTrain:
        ds = MultiThreadMapData(ds,nr_thread=16,map_func=data_prepare,buffer_size=200,strict=True)
        #ds = FakeData([[input_shape[0], input_shape[1], 3], [output_shape[0], output_shape[1],nr_skeleton]], 5000, random=False, dtype='uint8')
        ds = BatchData(ds, batch_size)
        ds = PrefetchDataZMQ(ds, 1)
    else:
        ds = MapData(ds, data_prepare)
    return ds



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
        return [tf.placeholder(tf.float32, [None, support_image_size[0], support_image_size[1], 3], 'first_image_masked'),
                #tf.placeholder(tf.float32, [None, support_image_size[0], support_image_size[1]], 'first_label'),
                tf.placeholder(tf.float32, [None, query_image_size[0], query_image_size[1], 3], 'second_image'),
                tf.placeholder(tf.int32, [None, query_image_size[0], query_image_size[1]], 'second_label')
                ]



    def build_graph(self, first_image_masked, second_image, second_label):
        first_image_masked = first_image_masked - tf.constant([104, 116, 122], dtype='float32')
        second_image = second_image - tf.constant([104, 116, 122], dtype='float32')

        ctx = get_current_tower_context()
        logger.info("current ctx.is_training: {}".format(ctx.is_training))

        with tf.variable_scope("support"):
             support_logits = deeplabv2(first_image_masked,CLASS_NUM,is_training=ctx.is_training)
        with tf.variable_scope("query"):
            query_logits = deeplabv2(second_image,CLASS_NUM,is_training=ctx.is_training)


        costs = []
        support_logits = tf.reduce_mean(support_logits, [1, 2], keep_dims=True, name='gap')
        support_logits = tf.image.resize_bilinear(support_logits, query_logits.shape[1:3])

        logits = support_logits + query_logits # 2048 channels

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
    global args
    logger.auto_set_dir(name=args.test_data)
    nr_tower = max(get_nr_gpu(), 1)
    total_batch = batch_size * nr_tower

    logger.info("Running on {} towers. Batch size per tower: {}".format(nr_tower, batch_size))
    dataset_train = get_data(args.train_data, batch_size)

    callbacks = [
        ModelSaver(),
        GPUUtilizationTracker(),
        EstimatedTimeLeft(),
        PeriodicTrigger(CalculateMIoU(CLASS_NUM), every_k_epochs=evaluate_every_n_epoch),
        ProgressBar(["cross_entropy_loss", "cost", "wd_cost"]) , # uncomment it to debug for every step
        #RunOp(lambda: tf.add_check_numerics_ops(), run_before=False, run_as_trigger=True, run_step=True)
    ]

    return TrainConfig(
        model=Model(),
        dataflow=dataset_train,
        callbacks=callbacks,
        steps_per_epoch=  10000// total_batch,
        max_epoch=max_epoch,
    )


class CalculateMIoU(Callback):
    def __init__(self, nb_class):
        self.nb_class = nb_class

    def _setup_graph(self):
        self.pred = self.trainer.get_predictor(
            ['first_image_masked','second_image'], ['prob'])

    def _before_train(self):
        pass

    def _trigger(self):
        global args
        self.val_ds = get_data(args.test_data)
        self.val_ds.reset_state()

        self.stat = MIoUStatistics(self.nb_class)

        for first_image_masks, second_image, second_label in tqdm(self.val_ds.get_data()):
            second_image = np.squeeze(second_image)
            second_label = np.squeeze(second_label)

            k_shot = len(first_image_masks)
            prediction_fused = np.zeros((support_image_size[0], support_image_size[1], CLASS_NUM), dtype=np.float32)
            for kk in range(k_shot):
                def mypredictor(input_img):
                    # input image: 1*H*W*3
                    # output : H*W*C
                    output = self.pred(first_image_masks[kk][np.newaxis, :, :, :], input_img)
                    return output[0][0]

                prediction = predict_scaler(second_image, mypredictor, scales=[0.5, 0.75, 1, 1.25, 1.5],
                                            classes=CLASS_NUM, tile_size=support_image_size, is_densecrf=False)
                prediction_fused += prediction

            prediction_fused = np.argmax(prediction_fused, axis=2)
            self.stat.feed(prediction_fused, second_label)

        logger.info("mIoU: {}".format(self.stat.mIoU))
        logger.info("mean_accuracy: {}".format(self.stat.mean_accuracy))
        logger.info("accuracy: {}".format(self.stat.accuracy))

def proceed_test(args, is_save = True):
    import cv2
    ds = get_data(args.test_data)


    result_dir = "result22"
    from tensorpack.utils.fs import mkdir_p
    mkdir_p(result_dir)


    pred_config = PredictConfig(
        model=Model(),
        session_init=my_get_model_loader(args.test_load),
        input_names=['first_image_masked','second_image'],
        output_names=['prob'])
    predictor = OfflinePredictor(pred_config)

    i = 0
    stat = MIoUStatistics(CLASS_NUM)
    logger.info("start validation....")
    for first_image_masks, second_image, second_label  in tqdm(ds.get_data()):
        second_image = np.squeeze(second_image)
        second_label = np.squeeze(second_label)

        k_shot = len(first_image_masks)
        prediction_fused = np.zeros((second_image.shape[0],second_image.shape[1],CLASS_NUM),dtype=np.float32)
        for kk in range(k_shot):
            def mypredictor(input_img):
                # input image: 1*H*W*3
                # output : H*W*C
                output = predictor(first_image_masks[kk][np.newaxis, :, :, :], input_img)
                return output[0][0]

            prediction = predict_scaler(second_image, mypredictor, scales=[0.5,0.75, 1, 1.25, 1.5], classes=CLASS_NUM, tile_size=support_image_size, is_densecrf = False)
            prediction_fused += prediction

        prediction_fused = np.argmax(prediction_fused, axis=2)
        stat.feed(prediction_fused, second_label)

        if is_save:
            cv2.imwrite("{}/{}.png".format(result_dir,i), np.concatenate((cv2.resize(first_image_masks[0],(second_image.shape[1],second_image.shape[0])),second_image, visualize_label(second_label), visualize_label(prediction_fused)), axis=1))

        i += 1

    logger.info("mIoU: {}".format(stat.mIoU))
    logger.info("mean_accuracy: {}".format(stat.mean_accuracy))
    logger.info("accuracy: {}".format(stat.accuracy))
    logger.info("mIoU beautify: {}".format(stat.mIoU_beautify))
    logger.info("matrix beatify: {}".format(stat.confusion_matrix_beautify))




def view(args):
    ds = RepeatedData(get_data('fold0_train'), -1)
    ds.reset_state()
    for inputs in ds.get_data():
        ##"""
        cv2.imshow("first_img_masked",(inputs[0][0]).astype(np.uint8))
        cv2.imshow("second_img", (inputs[1][0]).astype(np.uint8))
        cv2.imshow("second_label", visualize_label(inputs[2][0]))
        cv2.waitKey(10000)
        ##"""
        print "ssss"
        pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='5',help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load',default="slim_resnet_v2_101.ckpt", help='load model')
    parser.add_argument('--view', help='view dataset', action='store_true')
    parser.add_argument('--test_data', default="fold0_1shot_test", help='test data')
    parser.add_argument('--train_data', default="fold0_train", help='train data')
    parser.add_argument('--test', action='store_true', help='test data')
    parser.add_argument('--test_load', help='load model')

    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.view:
        view(args)
    elif args.test:
        assert args.test_load is not None
        proceed_test(args)
    else:
        config = get_config()
        if args.load:
            config.session_init = my_get_model_loader(args.load)


        nr_tower = max(get_nr_gpu(), 1)
        trainer = SyncMultiGPUTrainerReplicated(nr_tower)
        launch_train_with_config(config, trainer)
