# Author: Tao Hu <taohu620@gmail.com>
import tensorflow as tf

def softmax_cross_entropy_with_ignore_label(logits, label, class_num):
    """
    This function accepts logits rather than predictions, and is more numerically stable than
    :func:`class_balanced_cross_entropy`.
    logits: NxWxHxC
    labels: NxWxH
    """
    with tf.name_scope('softmax_cross_entropy_with_ignore_label'):
        #tf.assert_equal(logits.shape[1], label.shape[1])  # shape assert
        #TODO need assert here
        raw_prediction = tf.reshape(logits, [-1, class_num])
        label = tf.reshape(label,[-1,])
        #label_onehot = tf.one_hot(label, depth=class_num)
        indices = tf.squeeze(tf.where(tf.less(label, class_num)), axis=1)
        #raw_gt = tf.reshape(label_onehot, [-1, class_num])

        gt = tf.gather(label, indices)
        prediction = tf.gather(raw_prediction, indices)

        # Pixel-wise softmax loss.
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=gt)

        #TODO reduce_mean
    return loss

def soft_target_softmax_cross_entropy_with_ignore_label(logits, label, class_num):
    """
    This function accepts logits rather than predictions, and is more numerically stable than
    :func:`class_balanced_cross_entropy`.
    logits: NxWxHxC
    labels: NxWxH
    """
    with tf.name_scope('softmax_cross_entropy_with_ignore_label'):
        #tf.assert_equal(logits.shape[1], label.shape[1])  # shape assert
        #TODO need assert here
        raw_prediction = tf.reshape(logits, [-1, class_num])
        label = tf.reshape(label,[-1,])
        label_onehot = tf.one_hot(label, depth=class_num)

        indices = tf.squeeze(tf.where(tf.less(label, class_num)), axis=1)

        label_onehot = tf.where(tf.equal(label_onehot, 1), 0.9*tf.ones_like(label_onehot), 0.1*tf.ones_like(label_onehot)/(class_num-1))
        gt = tf.gather(label_onehot, indices)
        prediction = tf.gather(raw_prediction, indices)

        # Pixel-wise softmax loss.
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=gt)

        #TODO reduce_mean
    return loss


def online_bootstrapping(logits, label, class_num, pixels=1024):
    """
    This function accepts logits rather than predictions, and is more numerically stable than
    :func:`class_balanced_cross_entropy`.
    logits: NxWxHxC
    labels: NxWxH
    """
    with tf.name_scope('softmax_cross_entropy_with_ignore_label'):
        #tf.assert_equal(logits.shape[1], label.shape[1])  # shape assert
        #TODO need assert here
        raw_prediction = tf.reshape(logits, [-1, class_num])
        label = tf.reshape(label,[-1,])
        #label_onehot = tf.one_hot(label, depth=class_num)
        indices = tf.squeeze(tf.where(tf.less(label, class_num)), axis=1)
        #raw_gt = tf.reshape(label_onehot, [-1, class_num])

        gt = tf.gather(label, indices)
        prediction = tf.gather(raw_prediction, indices)


        gt_onehot = tf.one_hot(gt,depth=class_num)#NxC
        hardness = gt_onehot*prediction
        hardness = tf.reduce_sum(hardness,axis=1)#Nx1

        top_values, top_indices = tf.nn.top_k(hardness, sorted=False, k=pixels)
        gt = tf.gather(gt, top_indices)
        prediction = tf.gather(prediction, top_indices)


        # Pixel-wise softmax loss.
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=gt)
        loss = tf.reduce_mean(loss)
        #loss.set_shape([1])
    return loss



