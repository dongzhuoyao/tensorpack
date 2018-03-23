# Author: Tao Hu <taohu620@gmail.com>


import os
import gzip
import numpy as np
import cv2

from tensorpack.utils import logger
from tensorpack.dataflow.base import RNGDataFlow

from test import LoaderOfPairs, compute_net_inputs

__all__ = ['OneShotDataset']


class OneShotDataset(RNGDataFlow):
    def __init__(self,):
        pass

    def size(self):
        return 1e10

    @staticmethod
    def class_num():
        return 2

    def get_data(self): # only for one-shot learning
        settings = __import__('ss_settings')
        profile = getattr(settings, "fold0_1shot_test")
        loader = LoaderOfPairs(profile)

        loader.get_items_no_return()
        yield [loader.out['first_img'][0],loader.out['first_label'][0],loader.out['second_img'][0],loader.out['second_label'][0]]





if __name__ == '__main__':
    pass