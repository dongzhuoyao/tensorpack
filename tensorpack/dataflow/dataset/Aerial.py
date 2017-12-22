# Author: Tao Hu <taohu620@gmail.com>


import os
import gzip
import numpy as np
import cv2

from ...utils import logger
from ..base import RNGDataFlow

__all__ = ['Aerial']


class Aerial(RNGDataFlow):
    def __init__(self, meta_dir, name,
                 shuffle=None, dir_structure=None):

        assert name in ['train', 'val','test'], name
        assert os.path.isdir(meta_dir), meta_dir
        self.reset_state()
        self.name = name

        if shuffle is None:
            shuffle = name == 'train'
        self.shuffle = shuffle
        self.imglist = []

        if name == 'train':
            f = open(os.path.join(meta_dir,"train.txt"),"r")
        elif name == 'val':
            f = open(os.path.join(meta_dir, "val.txt"), "r")
        else:
            f = open(os.path.join(meta_dir, "test.txt"), "r")

        if name == "train" or name == "val":
            for line in f.readlines():
                self.imglist.append(line.strip("\n").split(" "))
            f.close()
        else:
            for line in f.readlines():
                self.imglist.append(line.strip("\n"))
            f.close()

        #self.imglist = self.imglist[:30]

    def size(self):
        return len(self.imglist)

    def get_data(self):
        idxs = np.arange(len(self.imglist))
        if self.shuffle:
            self.rng.shuffle(idxs)
        if  self.name == "train" or self.name == "val":
            for k in idxs:
                fpath, flabel = self.imglist[k]
                fpath = cv2.imread(fpath, cv2.IMREAD_COLOR)
                flabel = cv2.imread(flabel, cv2.IMREAD_GRAYSCALE)
                yield [fpath, flabel]
        else:
            for k in idxs:
                fpath = self.imglist[k]
                fname = os.path.basename(fpath).strip(".tiff")
                fpath = cv2.imread(fpath, cv2.IMREAD_COLOR)
                yield [fpath,fname]

if __name__ == '__main__':
    pass