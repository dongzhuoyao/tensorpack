import cv2
import tensorflow as tf
import argparse
from six.moves import zip
import os


from tensorpack import *
from tensorpack.dataflow import dataset
from tensorpack.utils.gpu import get_nr_gpu
from tensorpack.tfutils import optimizer
from tensorpack.tfutils.summary import add_moving_summary, add_param_summary
from tensorpack.utils import logger
import numpy as np
from eval_PCKh import pckh
from tqdm import tqdm
img_dir, meta_dir = "/data1/dataset/mpii/images", "metadata/mpii_annotations.json"
nr_skeleton = dataset.mpii.joint_num()
input_shape =(256, 256)
output_shape = (64, 64)

init_lr = 2.5e-4
lr_schedule = [(6, 1e-4), (9, 5e-5)]
max_epoch = 12
epoch_scale = 5 #10
evaluate_every_n_epoch = 1
stage = 4
batch_size = 16

from hg_model import make_network

class Model(ModelDesc):
    def _get_inputs(self):
        return [InputDesc(tf.float32, [None, input_shape[0], input_shape[1], 3], 'image'),
                InputDesc(tf.float32, [None, output_shape[0], output_shape[1], 16], 'heatmap')]

    def _build_graph(self, inputs):
        image,heatmap = inputs
        image = image - tf.constant([104, 116, 122], dtype='float32')
        ctx = get_current_tower_context()
        logger.info("current ctx.is_training: {}".format(ctx.is_training))
        predict, multi_stage_loss_dict = make_network(image, heatmap, stage, nr_skeleton, ctx.is_training)

        predict = tf.identity(predict,'predict')

        nodecay_loss = 0.
        for loss in multi_stage_loss_dict.values():
            nodecay_loss += loss / len(multi_stage_loss_dict)
        nodecay_loss = tf.identity(nodecay_loss, 'mse_loss')
        costs =[]
        costs.append(nodecay_loss)

        if get_current_tower_context().is_training:
            wd_w = tf.train.exponential_decay(2e-4, get_global_step_var(),
                                              80000, 0.7, True)
            wd_cost = tf.multiply(wd_w, regularize_cost('.*/weights', tf.nn.l2_loss), name='wd_cost')
            costs.append(wd_cost)
            self.cost = tf.add_n(costs, name='cost')


    def _get_optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=init_lr, trainable=False)
        opt = tf.train.AdamOptimizer(lr, epsilon=1e-3)
        return optimizer.apply_grad_processors(
            opt, [gradproc.ScaleGradient(
                [('nothing.*', 0.1), ('nothing.*', 5)])])


def get_data(name):
    global  args
    isTrain = name == 'train'
    ds = dataset.mpii(img_dir, meta_dir, name, input_shape, output_shape, shuffle=True)

    if isTrain:
        ds = BatchData(ds, args.batch_size)
        ds = PrefetchDataZMQ(ds, 1)
    else:
        ds = BatchData(ds, 1)
    return ds



def view_data():
    ds = RepeatedData(get_data('train'), -1)
    ds.reset_state()
    for image, heatmap  in ds.get_data():
        cv2.imshow("img",image[0])
        cv2.imshow("heatmap_gt",255*cv2.resize(np.sum(heatmap[0], axis=2),(input_shape[0],input_shape[1])))
        cv2.waitKey(4000)

class EvalPCKh(Callback):
    def __init__(self):
        pass

    def _setup_graph(self):
        self.pred = self.trainer.get_predictor(
            ['image'], ['predict'])

    def _before_train(self):
        pass

    def _trigger(self):
        global args
        origin_ds = ds = dataset.mpii(img_dir, meta_dir, "val", input_shape, output_shape, shuffle=False)
        #ds = BatchData(ds, 1)

        final_result = np.zeros((len(origin_ds.imglist), nr_skeleton, 2), np.float32)
        final_heatmap = np.zeros((len(origin_ds.imglist), nr_skeleton, output_shape[0], output_shape[1]), np.float32)
        image_id = 0
        _itr = ds.get_data()
        for _ in tqdm(range(len(origin_ds.imglist))):
            image, heatmap, meta = next(_itr)
            predicts = self.pred(image[None, :, :, :]) #[(8,1,64,64,16)]
            predict = predicts[0] # (8,1,64,64,16), 8 means stage, 1 means batch size(in prediction, batch size is 1)
            predict = np.squeeze(predict,axis=1) #reduce dimension 1
            predict = predict[-1, :, :, :]  # last stage
            final_heatmap[image_id, :, :, :] = np.transpose(predict, [2, 0, 1])
            if False:
                heatmap_view = np.sum(heatmap, axis=2)
                predict_view = np.sum(predict, axis=2)
                cv2.imshow("img", image)
                cv2.imshow("featmap", cv2.resize(heatmap_view, (input_shape[0], input_shape[1])))
                cv2.imshow("predict", cv2.resize(predict_view, (input_shape[0], input_shape[1])))
                cv2.waitKey()

            # TODO flip
            # TODO multi scale fusion
            for i in range(nr_skeleton):
                lb = predict[:, :, i].argmax()
                y, x = np.unravel_index(lb, predict[:, :, i].shape)  # notice the order of x,y
                final_result[image_id, i, 0] = x
                final_result[image_id, i, 1] = y

            final_result[image_id, :, 0] /= meta['transform']['divide_first'][0]
            final_result[image_id, :, 1] /= meta['transform']['divide_first'][1]
            final_result[image_id, :, 0] += meta['transform']['add_second'][0]
            final_result[image_id, :, 1] += meta['transform']['add_second'][1]

            # some coordinate may be negative after the transformation.
            img_height = meta['meta']['img_height']
            img_width = meta['meta']['img_width']
            final_result[image_id, :, 0] = np.minimum(final_result[image_id, :, 0], img_width)
            final_result[image_id, :, 0] = np.maximum(final_result[image_id, :, 0], 0)
            final_result[image_id, :, 1] = np.minimum(final_result[image_id, :, 1], img_height)
            final_result[image_id, :, 1] = np.maximum(final_result[image_id, :, 1], 0)

            if False:
                from utils import draw_skeleton
                big_img = cv2.imread(os.path.join(img_dir, meta['meta']['img_paths']))
                draw_skeleton(big_img, final_result[image_id])
                cv2.imshow("result", big_img)
                cv2.waitKey(3000)  # 3s

            image_id += 1

        pckh(final_result)

def proceed_validation(args, is_save = False):
    origin_ds = ds = dataset.mpii(img_dir, meta_dir, "val", input_shape, output_shape, shuffle=False)
    pred_config = PredictConfig(
        model=Model(),
        session_init=get_model_loader(args.load),
        input_names=['image'],
        output_names=['predict'])
    predictor = OfflinePredictor(pred_config)

    from tensorpack.utils.fs import mkdir_p
    result_dir = os.path.join("result_on_val")
    mkdir_p(result_dir)

    final_result = np.zeros((len(origin_ds.imglist), nr_skeleton, 2), np.float32)
    final_heatmap = np.zeros((len(origin_ds.imglist), nr_skeleton, output_shape[0], output_shape[1]), np.float32)
    image_id = 0
    _itr = ds.get_data()
    is_debug = False
    for _  in tqdm(range(len(origin_ds.imglist))):
        image, heatmap, meta = next(_itr)
        predicts = predictor(image[None, :, :, :])  # [(8,1,64,64,16)]
        predict = predicts[0]  # (8,1,64,64,16), 8 means stage, 1 means batch size(in prediction, batch size is 1)
        predict = np.squeeze(predict, axis=1)  # reduce dimension 1
        predict = predict[-1, :, :, :]  # last stage
        final_heatmap[image_id, :, :, :] = np.transpose(predict, [2, 0, 1])

        if is_debug:
            heatmap_view = np.sum(heatmap, axis=2)
            predict_view = np.sum(predict, axis=2)
            cv2.imshow("img", image)
            cv2.imshow("featmap", cv2.resize(heatmap_view, (input_shape[0], input_shape[1])))
            cv2.imshow("predict", cv2.resize(predict_view, (input_shape[0], input_shape[1])))
            #cv2.waitKey()

        # TODO flip
        # TODO multi scale fusion
        for i in range(nr_skeleton):
            lb = predict[:,:,i].argmax()
            y,x = np.unravel_index(lb, predict[:,:,i].shape) # notice the order of x,y
            final_result[image_id, i, 0] = x
            final_result[image_id, i, 1] = y

        final_result[image_id, :, 0] /= meta['transform']['divide_first'][0]
        final_result[image_id, :, 1] /= meta['transform']['divide_first'][1]
        final_result[image_id, :, 0] += meta['transform']['add_second'][0]
        final_result[image_id, :, 1] += meta['transform']['add_second'][1]

        # some coordinate may be negative after the transformation.
        img_height = meta['meta']['img_height']
        img_width = meta['meta']['img_width']
        final_result[image_id, :, 0] = np.minimum(final_result[image_id, :, 0],img_width)
        final_result[image_id, :, 0] = np.maximum(final_result[image_id, :, 0], 0)
        final_result[image_id, :, 1] = np.minimum(final_result[image_id, :, 1], img_height)
        final_result[image_id, :, 1] = np.maximum(final_result[image_id, :, 1], 0)


        if is_debug:
            from utils import draw_skeleton
            big_img = cv2.imread(os.path.join(img_dir,meta['meta']['img_paths']))
            draw_skeleton(big_img,final_result[image_id])
            cv2.imshow("result", big_img)
            cv2.waitKey(3000) #3s

        image_id += 1

    pckh(final_result)







def get_config():
    logger.auto_set_dir()
    dataset_train = get_data('train')
    steps_per_epoch = dataset_train.size() * epoch_scale
    return TrainConfig(
        dataflow=dataset_train,
        callbacks=[
            ModelSaver(),
            ScheduledHyperParamSetter('learning_rate', lr_schedule),
            HumanHyperParamSetter('learning_rate'),
            ProgressBar(['mse_loss', "cost", "wd_cost"]),  # uncomment it to debug for every step
            PeriodicTrigger(EvalPCKh(), every_k_epochs=evaluate_every_n_epoch),
        ],
        model=Model(),
        steps_per_epoch=steps_per_epoch,
        max_epoch=max_epoch,
    )



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu',default='2', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--batch_size', default=batch_size,type=int,  help='batch size')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--view', help='view dataset', action='store_true')
    parser.add_argument('--output', help='fused output filename. default to out-fused.png')
    parser.add_argument('--validation', action='store_true', help='validate model on validation images')
    parser.add_argument('--test', action='store_true', help='test model on test images')
    args = parser.parse_args()
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.view:
        view_data()
    elif args.validation:
        proceed_validation(args)
    else:
        config = get_config()
        if args.load:
            config.session_init = get_model_loader(args.load)
        launch_train_with_config(
            config,
            SyncMultiGPUTrainer(max(get_nr_gpu(), 1)))
