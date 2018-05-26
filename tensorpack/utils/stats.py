# -*- coding: UTF-8 -*-
# File: stats.py

import numpy as np
from ..utils.segmentation.segmentation import update_confusion_matrix
from .import logger
from numpy import linalg as LA

__all__ = ['StatCounter', 'BinaryStatistics', 'RatioCounter', 'Accuracy',
           'OnlineMoments', 'MIoUStatistics','MIoUBoundaryStatistics']


class StatCounter(object):
    """ A simple counter"""

    def __init__(self):
        self.reset()

    def feed(self, v):
        """
        Args:
            v(float or np.ndarray): has to be the same shape between calls.
        """
        self._values.append(v)

    def reset(self):
        self._values = []

    @property
    def count(self):
        return len(self._values)

    @property
    def average(self):
        assert len(self._values)
        return np.mean(self._values)

    @property
    def sum(self):
        assert len(self._values)
        return np.sum(self._values)

    @property
    def max(self):
        assert len(self._values)
        return max(self._values)

    @property
    def min(self):
        assert len(self._values)
        return min(self._values)


class RatioCounter(object):
    """ A counter to count ratio of something. """

    def __init__(self):
        self.reset()

    def reset(self):
        self._tot = 0
        self._cnt = 0

    def feed(self, cnt, tot=1):
        """
        Args:
            cnt(int): the count of some event of interest.
            tot(int): the total number of events.
        """
        self._tot += tot
        self._cnt += cnt

    @property
    def ratio(self):
        if self._tot == 0:
            return 0
        return self._cnt * 1.0 / self._tot

    @property
    def count(self):
        """
        Returns:
            int: the total
        """
        return self._tot


class Accuracy(RatioCounter):
    """ A RatioCounter with a fancy name """
    @property
    def accuracy(self):
        return self.ratio



class BinaryStatistics(object):
    """
    Statistics for binary decision,
    including precision, recall, false positive, false negative
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.nr_pos = 0  # positive label
        self.nr_neg = 0  # negative label
        self.nr_pred_pos = 0
        self.nr_pred_neg = 0
        self.corr_pos = 0   # correct predict positive
        self.corr_neg = 0   # correct predict negative

    def feed(self, pred, label):
        """
        Args:
            pred (np.ndarray): binary array.
            label (np.ndarray): binary array of the same size.
        """
        assert pred.shape == label.shape, "{} != {}".format(pred.shape, label.shape)
        self.nr_pos += (label == 1).sum()
        self.nr_neg += (label == 0).sum()
        self.nr_pred_pos += (pred == 1).sum()
        self.nr_pred_neg += (pred == 0).sum()
        self.corr_pos += ((pred == 1) & (pred == label)).sum()
        self.corr_neg += ((pred == 0) & (pred == label)).sum()

    @property
    def precision(self):
        if self.nr_pred_pos == 0:
            return 0
        return self.corr_pos * 1. / self.nr_pred_pos

    @property
    def recall(self):
        if self.nr_pos == 0:
            return 0
        return self.corr_pos * 1. / self.nr_pos

    @property
    def false_positive(self):
        if self.nr_pred_pos == 0:
            return 0
        return 1 - self.precision

    @property
    def false_negative(self):
        if self.nr_pos == 0:
            return 0
        return 1 - self.recall


class OnlineMoments(object):
    """Compute 1st and 2nd moments online (to avoid storing all elements).

    See algorithm at: https://www.wikiwand.com/en/Algorithms_for_calculating_variance#/Online_algorithm
    """

    def __init__(self):
        self._mean = 0
        self._M2 = 0
        self._n = 0

    def feed(self, x):
        """
        Args:
            x (float or np.ndarray): must have the same shape.
        """
        self._n += 1
        delta = x - self._mean
        self._mean += delta * (1.0 / self._n)
        delta2 = x - self._mean
        self._M2 += delta * delta2

    @property
    def mean(self):
        return self._mean

    @property
    def variance(self):
        return self._M2 / (self._n - 1)

    @property
    def std(self):
        return np.sqrt(self.variance)


class MIoUBoundaryStatistics(object):
    """
    Statistics for MIoUStatistics,
    including MIoU, accuracy, mean accuracy
    """
    def __init__(self, nb_classes, ignore_label=255, kernel = 7):
        self.nb_classes = nb_classes
        self.ignore_label = ignore_label
        self.kernel = kernel
        self.reset()

    def reset(self):
        self._mIoU = 0
        self._accuracy = 0
        self._mean_accuracy = 0
        self._confusion_matrix_boundary = np.zeros((self.nb_classes, self.nb_classes), dtype=np.uint64)
        self._confusion_matrix_inner = np.zeros((self.nb_classes, self.nb_classes), dtype=np.uint64)

    def distinguish_boundary(self,predict, label):
        w, h = label.shape
        r = self.kernel//2
        def is_boundary(gt, i, j):
            i_min = max(i-r,0)
            i_max = min(i+r+1,w)
            j_min = max(j-r,0)
            j_max = min(j+r+1,h)
            small_block = gt[i_min:i_max, j_min:j_max]

            if LA.norm(small_block - small_block[r,r],1) == 0: # if all element equal
                return False
            else:
                return True


        mask = np.zeros((w,h),dtype=np.uint8)
        for i in range(w):
            for j in range(h):
                mask[i, j] = is_boundary(label, i, j)
        boundary_idx = np.where(mask==1)
        inner_idx = np.where(mask==0)

        boundary_predict = predict[boundary_idx]
        inner_predict = predict[inner_idx]
        boundary_label = label[boundary_idx]
        inner_label = label[inner_idx]
        return boundary_predict,inner_predict,boundary_label,inner_label

    def feed(self, pred, label):
        """
        Args:
            pred (np.ndarray): binary array.
            label (np.ndarray): binary array of the same size.
        """
        assert pred.shape == label.shape, "{} != {}".format(pred.shape, label.shape)

        boundary_predict, inner_predict, boundary_label, inner_label = self.distinguish_boundary(pred,label)
        self._confusion_matrix_boundary = update_confusion_matrix(boundary_predict, boundary_label, self._confusion_matrix_boundary, self.nb_classes,
                                                         self.ignore_label)

        self._confusion_matrix_inner = update_confusion_matrix(inner_predict, inner_label,
                                                                  self._confusion_matrix_inner, self.nb_classes,
                                                                  self.ignore_label)

    @staticmethod
    def mIoU(_confusion_matrix):
        I = np.diag(_confusion_matrix)
        U = np.sum(_confusion_matrix, axis=0) + np.sum(_confusion_matrix, axis=1) - I
        assert np.min(U) > 0,"sample number is too small.."
        IOU = I*1.0 / U
        meanIOU = np.mean(IOU)
        return meanIOU

    @staticmethod
    def accuracy(_confusion_matrix):
        return np.sum(np.diag(_confusion_matrix))*1.0 / np.sum(_confusion_matrix)

    @staticmethod
    def mean_accuracy(_confusion_matrix):
        assert np.min(np.sum(_confusion_matrix, axis=1)) > 0, "sample number is too small.."
        return np.mean(np.diag(_confusion_matrix)*1.0 / np.sum(_confusion_matrix, axis=1))

    def print_result(self):
        logger.info("boundary result:")
        logger.info("boundary mIoU: {}".format(self.mIoU(self._confusion_matrix_boundary)))
        logger.info("boundary accuracy: {}".format(self.accuracy(self._confusion_matrix_boundary)))
        logger.info("boundary mean_accuracy: {}".format(self.mean_accuracy(self._confusion_matrix_boundary)))
        logger.info("inner result:")
        logger.info("inner mIoU: {}".format(self.mIoU(self._confusion_matrix_inner)))
        logger.info("inner accuracy: {}".format(self.accuracy(self._confusion_matrix_inner)))
        logger.info("inner mean_accuracy: {}".format(self.mean_accuracy(self._confusion_matrix_inner)))


class MIoUStatistics(object):
    """
    Statistics for MIoUStatistics,
    including MIoU, accuracy, mean accuracy
    """
    def __init__(self, nb_classes, ignore_label=255):
        self.nb_classes = nb_classes
        self.ignore_label = ignore_label
        self.reset()

    def reset(self):
        self._mIoU = 0
        self._accuracy = 0
        self._mean_accuracy = 0
        self._confusion_matrix = np.zeros((self.nb_classes, self.nb_classes), dtype=np.uint64)

    def feed(self, pred, label):
        """
        Args:
            pred (np.ndarray): binary array.
            label (np.ndarray): binary array of the same size.
        """
        assert pred.shape == label.shape, "{} != {}".format(pred.shape, label.shape)
        self._confusion_matrix = update_confusion_matrix(pred, label, self._confusion_matrix, self.nb_classes,
                                                         self.ignore_label)

    @property
    def confusion_matrix(self):
        return self._confusion_matrix

    @property
    def confusion_matrix_beautify(self):
        return np.array_str(self._confusion_matrix, precision=12, suppress_small=True)

    @property
    def IoU(self):
        I = np.diag(self._confusion_matrix)
        U = np.sum(self._confusion_matrix, axis=0) + np.sum(self._confusion_matrix, axis=1) - I
        # assert np.min(U) > 0,"sample number is too small.."
        IOU = I * 1.0 / U
        return IOU

    @property
    def mIoU(self):
        I = np.diag(self._confusion_matrix)
        U = np.sum(self._confusion_matrix, axis=0) + np.sum(self._confusion_matrix, axis=1) - I
        #assert np.min(U) > 0,"sample number is too small.."
        IOU = I*1.0 / U
        meanIOU = np.mean(IOU)
        return meanIOU

    @property
    def mIoU_beautify(self):
        I = np.diag(self._confusion_matrix)
        U = np.sum(self._confusion_matrix, axis=0) + np.sum(self._confusion_matrix, axis=1) - I
        #assert np.min(U) > 0, "sample number is too small.."
        IOU = I * 1.0 / U
        return np.array_str(IOU, precision=5, suppress_small=True)

    @property
    def accuracy(self):
        return np.sum(np.diag(self._confusion_matrix))*1.0 / np.sum(self._confusion_matrix)

    @property
    def mean_accuracy(self):
        #assert np.min(np.sum(self._confusion_matrix, axis=1)) > 0, "sample number is too small.."
        return np.mean(np.diag(self._confusion_matrix)*1.0 / np.sum(self._confusion_matrix, axis=1))



