# Author: Tao Hu <taohu620@gmail.com>


import os
import gzip
import numpy as np
import cv2

from tensorpack.utils import logger
from tensorpack.dataflow.base import RNGDataFlow
import ss_datalayer

__all__ = ['OneShotDatasetTwoBranch']


class OneShotDatasetTwoBranch(RNGDataFlow):
    def __init__(self,name):
        settings = __import__('ss_settings')
        self.name = name
        profile = getattr(settings, name)
        profile_copy = profile.copy()
        profile_copy['deploy_mode'] = True
        dbi = ss_datalayer.DBInterface(profile)
        self.data_size = len(dbi.db_items)
        self.PLP = ss_datalayer.PairLoaderProcess(dbi, profile_copy)

    def size(self):
        return self.data_size

    @staticmethod
    def class_num():
        return 2

    def get_data(self): # only for one-shot learning
        for i in range(self.data_size):
            first_index,second_index = self.PLP.load_next_frame() #like: [227],568
            yield [first_index, second_index]



if __name__ == '__main__':
    pass