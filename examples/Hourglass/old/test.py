import argparse
from config import config
import logging
from tqdm import tqdm
import cv2
import numpy as np
import argparse
import sys
sys.path.insert(0, '../../../data/MPIHP/')
from MPIAllJoints import MPIJoints
from dataset import preprocessing
from setproctitle import setproctitle
import tensorflow as tf
from hourglass import  make_network
from dataset import DataIter
import time,os
import ipdb
from visualization import draw_skeleton_new

logging.basicConfig(filename=config.link_log_file, filemode='a', level=logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
# create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# add formatter to ch
ch.setFormatter(formatter)
logging.getLogger().addHandler(ch)
run_mode = "test"

def get_arguments():
    """Parse all the arguments provided from the CLI.
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Hourglass Network")
    parser.add_argument("--batch_size", type=int, default=config.batch_size,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--restore_from", type=str,default=None,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--gpu", type=str, default="0",
                        help="Number of images sent to the network in one step.")
    return parser.parse_args()


def load(loader, sess, ckpt_path):
    '''Load trained weights.

    Args:
      loader: TensorFlow saver object.
      sess: TensorFlow session.
      ckpt_path: path to checkpoint file with parameters.
    '''
    loader.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))


def main():
    setproctitle(config.proj_name)
    args = get_arguments()
    with tf.device('/gpu:{}'.format(args.gpu)):
        nr_gpus = len(args.gpu.split(","))
        print("gpu nums: {}".format(nr_gpus))
        d = MPIJoints()
        test_data = d.load_realtest_data()
        test_data = test_data[:500]

        #ipdb.set_trace()
        output_num = 4
        pred_res = np.zeros((output_num, len(test_data), 16, 2))

        batch_size = 1
        metadatas = []

        # Create queue coordinator.
        coord = tf.train.Coordinator()

        data = tf.placeholder(tf.float32, shape=[batch_size*2, config.data_shape[0], config.data_shape[1], 3])
        label = tf.placeholder(tf.float32, shape=[batch_size*2,  config.output_shape[0],config.output_shape[1],config.nr_skeleton])

        # Create network.
        output, L = make_network(data,label, "train")
        predict = tf.nn.sigmoid(output)

        # Set up tf session and initialize variables.
        tf_config = tf.ConfigProto(allow_soft_placement = True)
        tf_config.gpu_options.allow_growth = True
        sess = tf.Session(config=tf_config)

        init = tf.initialize_all_variables()
        sess.run(init)


        # Saver for storing checkpoints of the model.
        saver = tf.train.Saver(var_list=tf.all_variables(), max_to_keep=40)

        if args.restore_from is not None:
            load(saver, sess, args.restore_from)

        # Start queue threads.
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)

        for test_id in tqdm(range(0, len(test_data), batch_size)):
            start_id = test_id
            end_id = min(len(test_data), test_id + batch_size)

            test_img, test_label,metadata = preprocessing(test_data[start_id:end_id], shape=config.data_shape)

            feed = test_img.copy()
            for i in range(end_id - start_id):
                ori_img = test_img[i].transpose(1, 2, 0)
                flip_img = cv2.flip(ori_img, 1)
                feed = np.vstack((feed, flip_img.transpose(2, 0, 1)[np.newaxis, ...]))

            feed = np.transpose(feed,(0,2,3,1))

            res_ori = sess.run([predict],
                                       feed_dict={data: feed.astype(np.float32)})


            #ipdb.set_trace()

            res_ori = res_ori[0]
            output_num = len(res_ori)
            for lab in range(output_num):
                res = np.array(res_ori[lab])
                res = np.transpose(res,(0,3,1,2))#ht

                #deal with flipped part
                for i in range(end_id - start_id):
                    fmp = res[end_id - start_id + i].transpose((1, 2, 0))
                    fmp = cv2.flip(fmp, 1);
                    fmp = list(fmp.transpose((2, 0, 1)))
                    # symmetry = [ (1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16) ]
                    symmetry = [(0, 5), (1, 4), (2, 3), (10, 15), (11, 14), (12, 13)]
                    for (q, w) in symmetry:
                        fmp[q], fmp[w] = fmp[w], fmp[q];
                    fmp = np.array(fmp);
                    res[i] += fmp;
                    res[i] /= 2;

                for test_image_id in range(start_id, end_id):
                    # if test_image_id % 100 == 0 :
                    #    print(test_image_id)
                    for w in range(16):
                        res[test_image_id - start_id, w] /= np.amax(res[test_image_id - start_id, w])
                    sk = [(0, 0) for i in range(16)]

                    border = 10
                    dr = np.zeros((16, config.output_shape[0] + 2 * border, config.output_shape[1] + 2 * border))
                    dr[:, border:-border, border:-border] = res[test_image_id - start_id].copy()

                    for w in range(16):
                        dr[w] = cv2.GaussianBlur(dr[w], (21, 21), 0)


                    for w in range(16):##??
                        #ipdb.set_trace()
                        lb = dr[w].argmax()
                        x, y = np.unravel_index(lb, dr[w].shape)
                        dr[w, x, y] = 0
                        lb = dr[w].argmax()
                        px, py = np.unravel_index(lb, dr[w].shape)
                        x -= border
                        y -= border
                        px -= border + x
                        py -= border + y
                        ln = (px ** 2 + py ** 2) ** 0.5
                        delta = 0.25
                        if ln > 1e-3:
                            x += delta * px / ln
                            y += delta * py / ln
                        sk[w] = (y * 4 + 2, x * 4 + 2)
                    pred_res[lab][test_image_id] = sk

            #ipdb.set_trace()

            metadatas.extend(metadata)
        coord.request_stop()
        coord.join(threads)

        start_epoch = int(args.restore_from.split("-")[1])
        generate_visualization_and_txt(test_data, pred_res[-1].copy(), metadatas, start_epoch)


def ensure_dir(path):
    """create directories if *path* does not exist"""
    if not os.path.isdir(path):
        os.makedirs(path)


def generate_visualization_and_txt(test_data, pred, metadatas, epoch,save_img=True):
    log_folder = os.path.join(config.log_folder, "{}-{}-epoch{}".format(config.this_dir,run_mode,epoch))
    ensure_dir(log_folder)
    f = open(os.path.join(log_folder,"00000000test_result.txt"),"w")

    for i in tqdm(range(len(test_data))):
        smallimg, _, _ = preprocessing([test_data[i]], shape=config.data_shape, return_patch_for_check=True)
        smallimg = smallimg[0]
        smallimg = np.transpose(smallimg, (1,2,0))

        smallimg_ =smallimg
        imgname = test_data[i]["imgpath"]
        indexid = test_data[i]["indexid"]
        personid = test_data[i]["personid"]
        small_label = pred[i].copy()
        label = pred[i]
        metadata = metadatas[i]
        f.write("{},{},{}".format(imgname,indexid,personid))
        for j in range(config.nr_skeleton):
            label[j][0] = label[j][0]/metadata["x_ratio"]
            label[j][1] = label[j][1] / metadata["y_ratio"]
            label[j][0] = label[j][0] + metadata["min_x"]
            label[j][1] = label[j][1] + metadata["min_y"]

            f.write(",{},{}".format(label[j][0],label[j][1]))
        f.write("\n")

        if save_img:
            img = cv2.imread(os.path.join(config.img_path, imgname))
            img = draw_skeleton_new(img, label.astype(int))
            #ipdb.set_trace()
            smallimg_ = draw_skeleton_new(smallimg_,small_label.astype(int))
            # cv2.rectangle(img, (headRect[0],headRect[1]), (headRect[2], headRect[3]), (0, 255,0), 3)
            cv2.imwrite(os.path.join(log_folder,"img{}-index{}-person{}-epoch{}.jpg".format(imgname.replace(".jpg",""),indexid,personid,epoch))
                                     , img)
            cv2.imwrite(os.path.join(log_folder,
                                     "img{}-index{}-person{}-patch-epoch{}.jpg".format(imgname.replace(".jpg", ""), indexid,
                                                                                 personid, epoch))
                        , smallimg_)

if __name__ == '__main__':
    main()
