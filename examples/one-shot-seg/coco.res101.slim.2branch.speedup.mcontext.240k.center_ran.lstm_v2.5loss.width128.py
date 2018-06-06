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
from coco_dataset_split5.OneShotDatasetTwoBranch_Coco import OneShotDatasetTwoBranchCoco
from deeplabv2_dilation6_new_mtscale import deeplabv2
import tensorflow as tf
image_size = (320, 320)
slim = tf.contrib.slim
from RAN import AttentionModule
max_epoch = 6
weight_decay = 5e-4

LR = 1e-4
CLASS_NUM = 2
evaluate_every_n_epoch = 1
support_image_size =image_size
query_image_size = image_size
images_per_epoch =40000
fusion_width = 256
lstm_mid_channel = 128

batch_size = 5
k_shot = 5

from cell import ConvLSTMCell_carlthome
"""
python coco.res101.slim.2branch.speedup.mcontext.240k.center_ran.lstm_v2.5loss.width128.py --test_load train_log/coco.res101.slim.2branch.speedup.mcontext.240k.center_ran.lstm_v2.5loss.width128:fold0_5shot_test/model-24000 --k_shot 1 --test --test_data fold0_1shot_test --gpu 4 &&
python  coco.res101.slim.2branch.speedup.mcontext.240k.center_ran.lstm_v2.5loss.width128.py --test_load   train_log/coco.res101.slim.2branch.speedup.mcontext.240k.center_ran.lstm_v2.5loss.width128:fold1_5shot_test/model-24000 --k_shot 1 --test --test_data fold1_1shot_test --gpu 4 &&
python  coco.res101.slim.2branch.speedup.mcontext.240k.center_ran.lstm_v2.5loss.width128.py --test_load   train_log/coco.res101.slim.2branch.speedup.mcontext.240k.center_ran.lstm_v2.5loss.width128:fold2_5shot_test/model-8000 --k_shot 1 --test --test_data fold2_1shot_test --gpu 4 &&
python  coco.res101.slim.2branch.speedup.mcontext.240k.center_ran.lstm_v2.5loss.width128.py --test_load   train_log/coco.res101.slim.2branch.speedup.mcontext.240k.center_ran.lstm_v2.5loss.width128:fold3_5shot_test/model-16000 --k_shot 1 --test --test_data fold3_1shot_test --gpu 4 

"""


#  python res101.slim.2branch.speedup.mcontext.240k.center_ran.lstm_v2.5loss.width128.py --test_load train_log/res101.slim.2branch.speedup.mcontext.240k.center_ran.lstm_v2.5loss.width128:fold0_5shot_test/model-48000 --k_shot 2 --test --test_data fold0_2shot_test --gpu 2
#  python res101.slim.2branch.speedup.mcontext.240k.center_ran.lstm_v2.5loss.width128.py --test_load train_log/res101.slim.2branch.speedup.mcontext.240k.center_ran.lstm_v2.5loss.width128:fold0_5shot_test/model-48000 --k_shot 3 --test --test_data fold0_3shot_test --gpu 2
#  python res101.slim.2branch.speedup.mcontext.240k.center_ran.lstm_v2.5loss.width128.py --test_load train_log/res101.slim.2branch.speedup.mcontext.240k.center_ran.lstm_v2.5loss.width128:fold0_5shot_test/model-48000 --k_shot 4 --test --test_data fold0_4shot_test --gpu 2

#  python res101.slim.2branch.speedup.mcontext.240k.center_ran.lstm_v2.5loss.width128.py --test_load train_log/res101.slim.2branch.speedup.mcontext.240k.center_ran.lstm_v2.5loss.width128:fold0_5shot_test/model-48000 --k_shot 6 --test --test_data fold0_6shot_test --gpu 2
#  python res101.slim.2branch.speedup.mcontext.240k.center_ran.lstm_v2.5loss.width128.py --test_load train_log/res101.slim.2branch.speedup.mcontext.240k.center_ran.lstm_v2.5loss.width128:fold0_5shot_test/model-48000 --k_shot 7 --test --test_data fold0_7shot_test --gpu 2
#  python res101.slim.2branch.speedup.mcontext.240k.center_ran.lstm_v2.5loss.width128.py --test_load train_log/res101.slim.2branch.speedup.mcontext.240k.center_ran.lstm_v2.5loss.width128:fold0_5shot_test/model-48000 --k_shot 8 --test --test_data fold0_8shot_test --gpu 2
#  python res101.slim.2branch.speedup.mcontext.240k.center_ran.lstm_v2.5loss.width128.py --test_load train_log/res101.slim.2branch.speedup.mcontext.240k.center_ran.lstm_v2.5loss.width128:fold0_5shot_test/model-48000 --k_shot 9 --test --test_data fold0_9shot_test --gpu 2
#  python res101.slim.2branch.speedup.mcontext.240k.center_ran.lstm_v2.5loss.width128.py --test_load train_log/res101.slim.2branch.speedup.mcontext.240k.center_ran.lstm_v2.5loss.width128:fold0_5shot_test/model-48000 --k_shot 10 --test --test_data fold0_10shot_test --gpu 2

def my_squeeze_excitation_layer(input_x, out_dim, layer_name,ratio=4):
  with tf.variable_scope(layer_name):
    squeeze = tf.reduce_mean(input_x, [1, 2], name='gap', keep_dims=False)

    with tf.variable_scope('fc1'):
      excitation = tf.layers.dense(inputs=squeeze, use_bias=True, units=int(out_dim / ratio))

    excitation = tf.nn.relu(excitation)

    with tf.variable_scope('fc2'):
      excitation = tf.layers.dense(inputs=excitation, use_bias=True, units=out_dim)
    excitation = tf.nn.sigmoid(excitation)

    excitation = tf.reshape(excitation, [-1, 1, 1, out_dim])
    scale = input_x * excitation
    return scale

def get_data(name,batch_size=1):
    isTrain = True if 'train' in name else False
    ds = OneShotDatasetTwoBranchCoco(name, image_size=support_image_size)


    if isTrain:
        ds = BatchData(ds, batch_size)
        ds = PrefetchDataZMQ(ds, 1)
    else:
        ds = BatchData(ds, 1)
        #ds = PrefetchDataZMQ(ds, 1)
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
        global args
        return [tf.placeholder(tf.float32, [None,args.k_shot, support_image_size[0], support_image_size[1], 3], 'first_image_masked'),
                #tf.placeholder(tf.float32, [None, support_image_size[0], support_image_size[1]], 'first_label'),
                tf.placeholder(tf.float32, [None, query_image_size[0], query_image_size[1], 3], 'second_image'),
                tf.placeholder(tf.int32, [None, query_image_size[0], query_image_size[1]], 'second_label')
                ]



    def build_graph(self, first_image_masked, second_image, second_label):
        #first_image_masked = first_image_masked - tf.constant([104, 116, 122], dtype='float32')
        second_image = second_image - tf.constant([104, 116, 122], dtype='float32')

        ctx = get_current_tower_context()
        logger.info("current ctx.is_training: {}".format(ctx.is_training))


        cell = ConvLSTMCell_carlthome([20, 20], filters=lstm_mid_channel, kernel = [3, 3],reuse=tf.AUTO_REUSE)



        first_image_masked = tf.reshape(first_image_masked,(-1,image_size[0],image_size[1],3))

        first_image_masked = first_image_masked - tf.constant([104, 116, 122], dtype='float32')
        with tf.variable_scope("support"):
             support_context_list = deeplabv2(first_image_masked,CLASS_NUM,is_training=ctx.is_training)
        with tf.variable_scope("query"):
            query_context_list = deeplabv2(second_image,CLASS_NUM,is_training=ctx.is_training)

        def smooth(inp, conv_width, name, stride=1,output_num=fusion_width):
            with tf.variable_scope(name):
                return slim.conv2d(inp, output_num, [conv_width, conv_width], stride=stride,
                                        activation_fn=None, normalizer_fn=None)

        for iii in range(len(support_context_list)):
            shape_list = support_context_list[iii].get_shape().as_list()
            support_context_list[iii] = tf.reshape(support_context_list[iii],(-1,args.k_shot,shape_list[1],shape_list[2],shape_list[3]))

        fusion_list = []
        final_list = []
        with tf.variable_scope('') as scope:
            for kth_shot in range(args.k_shot):
                if kth_shot > 0:
                    scope.reuse_variables()
                fusion_branch = smooth(support_context_list[0][:,kth_shot,:,:,:],1,"context_support0")+ \
                                smooth(query_context_list[0], 1, "context_query0")

                fusion_branch = AttentionModule(fusion_branch, fusion_width, "center0_ran")

                fusion_branch = smooth(fusion_branch,1,"context_fusion0",stride=2)


                fusion_branch = fusion_branch + \
                                smooth(support_context_list[1][:,kth_shot,:,:,:], 1, "context_support1") + \
                                smooth(query_context_list[1], 1, "context_query1")

                fusion_branch = AttentionModule(fusion_branch, fusion_width, "center1_ran")

                fusion_branch = smooth(fusion_branch, 1, "context_fusion1", stride=1)



                fusion_branch = fusion_branch + \
                                smooth(support_context_list[2][:,kth_shot,:,:,:], 1, "context_support2") + \
                                smooth(query_context_list[2], 1, "context_query2")

                fusion_branch = AttentionModule(fusion_branch, fusion_width, "center2_ran")

                fusion_branch = smooth(fusion_branch, 1, "context_fusion2", stride=1)



                fusion_branch = fusion_branch + \
                                smooth(support_context_list[3][:,kth_shot,:,:,:], 1, "context_support3") + \
                                smooth(query_context_list[3], 1, "context_query3")
                fusion_branch = smooth(fusion_branch, 1, "context_fusion3", stride=1, output_num=lstm_mid_channel) # [batch_size,w,h,c]
                fusion_list.append(fusion_branch)

            fusion_branch = tf.stack(fusion_list)
            fusion_branch = tf.transpose(fusion_branch,(1, 0, 2, 3, 4))# batch_size, time_step, w, h, c

            fusion_branch, state = tf.nn.dynamic_rnn(cell, fusion_branch, dtype=fusion_branch.dtype)

            fusion_branch = tf.split(fusion_branch,axis=1,num_or_size_splits=args.k_shot)

        with tf.variable_scope('') as scope:
            for iii in range(args.k_shot):
                    if iii > 0:
                        scope.reuse_variables()
                    final_list.append(smooth(tf.squeeze(fusion_branch[iii],axis=1), 1, "context_after_lstm", stride=1, output_num=CLASS_NUM))


        costs = []
        logits = tf.image.resize_bilinear(final_list[-1], second_image.shape[1:3],name="upsample")
        prob = tf.nn.softmax(logits, name='prob')

        if get_current_tower_context().is_training:
            for jjj in range(args.k_shot):
                logits = tf.image.resize_bilinear(final_list[jjj], second_image.shape[1:3])
                cost = softmax_cross_entropy_with_ignore_label(logits, second_label, class_num=CLASS_NUM)
                cost = tf.reduce_mean(cost, name='cross_entropy_loss')/args.k_shot
                costs.append(cost)

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
        ProgressBar(["cross_entropy_loss", "cost", "learning_rate"]),  # uncomment it to debug for every step
        RunOp(lambda: tf.group(get_global_step_var().assign(0)), run_before=True, run_as_trigger=False, run_step=False,
              verbose=True)
    ]

    return TrainConfig(
        model=Model(),
        dataflow=dataset_train,
        callbacks=callbacks,
        steps_per_epoch=  images_per_epoch// total_batch,
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

        for first_image_masks, second_image, second_label in tqdm(self.val_ds.get_data(), total=self.val_ds.size()):
            second_image = np.squeeze(second_image)
            second_label = np.squeeze(second_label)
            first_image_masks = np.squeeze(first_image_masks)

            if len(first_image_masks.shape) == 3:# make 1-shot runnable
                first_image_masks = first_image_masks[np.newaxis,:, :, :]

            prediction_fused = np.zeros((second_image.shape[0], second_image.shape[1], CLASS_NUM), dtype=np.float32)
            def mypredictor(input_img):
                # input image: 1*H*W*3
                # output : H*W*C
                output = self.pred(first_image_masks[np.newaxis,:, :, :, :], input_img[np.newaxis,:, :, :])
                return output[0][0]

            prediction = predict_scaler(second_image, mypredictor, scales=[1],
                                        classes=CLASS_NUM, tile_size=support_image_size, is_densecrf=False)
            prediction_fused += prediction

            prediction_fused = np.argmax(prediction_fused, axis=2)
            self.stat.feed(prediction_fused, second_label)



        logger.info("mIoU: {}".format(self.stat.mIoU))
        logger.info("mean_accuracy: {}".format(self.stat.mean_accuracy))
        logger.info("accuracy: {}".format(self.stat.accuracy))
        logger.info("mIoU beautify: {}".format(self.stat.mIoU_beautify))
        logger.info("matrix beatify: {}".format(self.stat.confusion_matrix_beautify))

def proceed_test(args, is_save = False):
    import cv2
    ds = get_data(args.test_data)


    result_dir = "result22"
    from tensorpack.utils.fs import mkdir_p
    mkdir_p(result_dir)


    pred_config = PredictConfig(
        model=Model(),
        session_init=get_model_loader(args.test_load),
        input_names=['first_image_masked','second_image'],
        output_names=['prob'])
    predictor = OfflinePredictor(pred_config)

    i = 0
    stat = MIoUStatistics(CLASS_NUM)
    logger.info("start validation....")
    for first_image_masks, second_image, second_label  in tqdm(ds.get_data(),total=ds.size()):
        second_image = np.squeeze(second_image)
        second_label = np.squeeze(second_label)
        first_image_masks = np.squeeze(first_image_masks)

        if len(first_image_masks.shape) == 3:  # make 1-shot runnable
            first_image_masks = first_image_masks[np.newaxis, :, :, :]

        prediction_fused = np.zeros((second_image.shape[0], second_image.shape[1], CLASS_NUM), dtype=np.float32)

        def mypredictor(input_img):
            # input image: 1*H*W*3
            # output : H*W*C
            output = predictor(first_image_masks[np.newaxis, :, :, :, :], input_img[np.newaxis,:, :, :])
            return output[0][0]

        prediction = predict_scaler(second_image, mypredictor, scales=[1],
                                    classes=CLASS_NUM, tile_size=support_image_size, is_densecrf=False)
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
    ds = RepeatedData(get_data('fold0_5shot_train'), -1)
    ds.reset_state()
    for inputs in ds.get_data():
        ##"""
        cv2.imshow("first_img_masked0",(inputs[0][0,0]).astype(np.uint8))
        cv2.imshow("first_img_masked1", (inputs[0][0,1]).astype(np.uint8))
        cv2.imshow("first_img_masked2", (inputs[0][0,2]).astype(np.uint8))
        cv2.imshow("first_img_masked3", (inputs[0][0,3]).astype(np.uint8))
        cv2.imshow("first_img_masked4", (inputs[0][0,4]).astype(np.uint8))
        cv2.imshow("second_img", (inputs[1][0]).astype(np.uint8))
        cv2.imshow("second_label", visualize_label(inputs[2][0]))
        cv2.waitKey(10000)
        ##"""
        print "ssss"
        pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='1',help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load',default="slim_resnet_v2_101.ckpt", help='load model')
    parser.add_argument('--view', help='view dataset', action='store_true')
    parser.add_argument('--test_data', default="fold0_5shot_test", help='test data')
    parser.add_argument('--train_data', default="fold0_5shot_train", help='train data')
    parser.add_argument('--k_shot', default=5, type=int, help='k_shot')
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
            from sess_utils import my_get_model_loader
            config.session_init = my_get_model_loader(args.load)


        nr_tower = max(get_nr_gpu(), 1)
        trainer = SyncMultiGPUTrainerReplicated(nr_tower)
        launch_train_with_config(config, trainer)