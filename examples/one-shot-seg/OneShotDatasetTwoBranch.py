# Author: Tao Hu <taohu620@gmail.com>


import os
import gzip
import numpy as np
import cv2

from tensorpack.utils import logger
from tensorpack.dataflow.base import RNGDataFlow

from test import LoaderOfPairs

__all__ = ['OneShotDatasetTwoBranch']


class OneShotDatasetTwoBranch(RNGDataFlow):
    def __init__(self,name):
        settings = __import__('ss_settings')
        self.name = name
        profile = getattr(settings, name)
        self.loader = LoaderOfPairs(profile)
        self.data_size = self.loader.data_size

    def size(self):
        return self.data_size

    @staticmethod
    def class_num():
        return 2

    def get_data(self): # only for one-shot learning
        for i in range(self.data_size):
            self.loader.get_items_no_return()
            first_image = self.loader.out['first_img'][0]
            first_label = self.loader.out['first_label'][0]
            first_image_masked = first_image*first_label[:,:,np.newaxis]
            yield [first_image_masked, self.loader.out['second_img'][0],self.loader.out['second_label'][0]]



if __name__ == '__main__':
    pass