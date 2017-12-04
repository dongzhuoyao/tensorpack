# Author: Tao Hu <taohu620@gmail.com>


import os
import gzip
import numpy as np
import cv2

from ...utils import logger
from ..base import RNGDataFlow

__all__ = ['NingboEdge']


class NingboEdge(RNGDataFlow):
    def __init__(self, dir, meta_dir, edge_dir, name,
                 shuffle=None, dir_structure=None):

        assert name in ['train', 'val'], name
        assert os.path.isdir(dir), dir
        self.dir = dir
        self.edge_dir = edge_dir
        self.name = name

        if shuffle is None:
            shuffle = name == 'train'
        self.shuffle = shuffle
        self.imglist = []

        if name == 'train':
            f = open(os.path.join(meta_dir,"train.txt"),"r")
        else:
            f = open(os.path.join(meta_dir, "val.txt"), "r")

        for line in f.readlines():
            self.imglist.append(line.strip("\n").split(" "))
        f.close()

        #self.imglist = self.imglist[:80]

    def size(self):
        return len(self.imglist)

    def get_data(self):
        idxs = np.arange(len(self.imglist))
        if self.shuffle:
            self.rng.shuffle(idxs)

        for k in idxs:
            fname, flabel = self.imglist[k]
            fname_path = os.path.join(self.dir, fname)
            fedgemap_path = os.path.join(self.edge_dir,flabel)
            fname = cv2.imread(fname_path, cv2.IMREAD_COLOR)
            fedgemap = cv2.imread(fedgemap_path, cv2.IMREAD_GRAYSCALE)
            yield [fname,fedgemap]




if __name__ == '__main__':
    pass