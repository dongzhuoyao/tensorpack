# Author: Tao Hu <taohu620@gmail.com>
import tensorflow as tf

def kl_loss_compute(logits1, logits2, name):
    """ KL loss
    """
    pred1 = tf.nn.softmax(logits1)
    pred2 = tf.nn.softmax(logits2)
    loss = tf.reduce_mean(tf.reduce_sum(pred2 * tf.log(1e-8 + pred2 / (pred1 + 1e-8)), 1),name=name)

    return loss