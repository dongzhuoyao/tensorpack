#!/usr/bin/python3
import argparse
from config import config
import logging
import os
import time
import sys
import tensorflow as tf
from hourglass import  make_network
from setproctitle import setproctitle
from dataset import DataIter
sys.path.insert(0, '../../../data/MPIHP/')
from MPIAllJoints import MPIJoints


logging.basicConfig(filename=config.link_log_file, filemode='a', level=logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
# create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# add formatter to ch
ch.setFormatter(formatter)
logging.getLogger().addHandler(ch)


def link_file(src, target):
    if os.path.isdir(target) or os.path.isfile(target):
        os.remove(target)
    os.system('ln -s {} {}'.format(src, target))


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


def save(saver, sess, logdir, step):
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    saver.save(sess, checkpoint_path, global_step=step)
    print('The checkpoint has been created.')


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
        train_data, validation_data = d.load_data()

        # Create queue coordinator.
        coord = tf.train.Coordinator()


        data = tf.placeholder(tf.float32, shape=[config.batch_size, config.data_shape[0], config.data_shape[1], 3])
        label = tf.placeholder(tf.float32, shape=[config.batch_size,  config.output_shape[0],config.output_shape[1],config.nr_skeleton])
        learning_rate = tf.placeholder(tf.float32, shape=[])

        # Create network.
        output, L = make_network(data,label, "train")

        # Define the loss and optimisation parameters.
        loss = L["total_loss"]
        optimiser = tf.train.AdamOptimizer(learning_rate)

        trainable = tf.trainable_variables()
        optim = optimiser.minimize(loss, var_list=trainable)

        # Set up tf session and initialize variables.
        tf_config = tf.ConfigProto(allow_soft_placement = True)
        tf_config.gpu_options.allow_growth = True
        sess = tf.Session(config=tf_config)

        tf.summary.scalar('mse0', L["mse0"])
        tf.summary.scalar('mse1', L["mse1"])
        tf.summary.scalar('mse2', L["mse2"])
        tf.summary.scalar('mse3', L["mse3"])
        tf.summary.scalar('total_loss', loss)
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(config.tensorboard_logger_log_dir, sess.graph)

        init = tf.initialize_all_variables()

        sess.run(init)

        # Saver for storing checkpoints of the model.
        saver = tf.train.Saver(var_list=trainable)
        writeSaver = tf.train.Saver(max_to_keep=1000)
        start_epoch = -1
        if args.restore_from is not None:
            start_epoch = int(args.restore_from.split("-")[1])
            print("new start epoch: {}".format(start_epoch))
            load(saver, sess, args.restore_from)

        # Start queue threads.
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)


        for  cur_epoch in range(start_epoch+1,config.nr_epochs):
            bb = cur_epoch/config.lr_dec_epoch
            cur_lr = config.lr*(config.lr_gamma)**bb

            epoch_start_time = time.time()
            start_time = time.time()
            idx = 0
            di = DataIter("train", train_data, config.data_shape)
            while True:
                minibatch = next(di)
                iter_start_time = time.time()
                loss_value, _ = sess.run([loss, optim],feed_dict= {data: minibatch[0], label: minibatch[1],learning_rate:cur_lr})
                iter_cost_time = time.time() - iter_start_time
                epoch_elapsed_time = time.time() - epoch_start_time

                nr_iters = (idx + 1) * nr_gpus
                iter_avg_cost_time = epoch_elapsed_time / nr_iters
                nr_imgs = nr_iters * config.batch_size
                img_avg_cost_time = epoch_elapsed_time / nr_imgs

                epoch_expected_cost_time = img_avg_cost_time * config.epoch_size
                past_iterations = cur_epoch * config.epoch_size
                cur_iteration = past_iterations + nr_imgs

                outputs = [
                    'exp-name: {}'.format(config.exp_name),
                    'gpus: {}'.format(args.gpu),
                    'epoch-{}:'.format(cur_epoch),
                    'train_loss=%.7f' % loss_value,
                    '%.2fs/iter' % (iter_cost_time),
                    'ACC %.2fs/iter' % (iter_avg_cost_time),
                    'done=%d/%d' % (nr_imgs, config.epoch_size),
                    'elapsed_time=%.1f' % (epoch_elapsed_time),
                    'total_time=%.1f' % (epoch_expected_cost_time),
                    'cur_iteration={}'.format(cur_iteration),
                    'lr=%.1e' % (cur_lr),
                ]

                logging.info(' '.join(outputs))
                if idx % 50 == 0:
                    merged_summary = sess.run(merged,feed_dict= {data: minibatch[0], label: minibatch[1],learning_rate:cur_lr})
                    train_writer.add_summary(merged_summary, cur_iteration)
                    train_writer.flush()

                if nr_imgs > config.epoch_size:
                    break

                idx += 1

            elapsed_time = time.time() - start_time
            if cur_epoch % 1 == 0:
                save(writeSaver, sess, config.output_dir, cur_epoch)

        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    main()
