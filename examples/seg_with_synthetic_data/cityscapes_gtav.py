# Author: Tao Hu <taohu620@gmail.com>


import os
import gzip
import numpy as np
import cv2

from tensorpack.utils import logger
from tensorpack.dataflow.base import RNGDataFlow
from PIL import Image
__all__ = ['Cityscapes_GTAV']


id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                              26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}

root_gtav = "./data/GTA5"
gtav_train_list= "metadata/gta5_list/train.txt"


class Cityscapes_GTAV(RNGDataFlow):
    def __init__(self, root_dir, meta_dir, name,
                 partial_data = -1, shuffle=None, dir_structure=None):

        assert name in ['train', 'val','test'], name
        assert os.path.isdir(meta_dir), meta_dir
        assert os.path.isdir(root_dir), root_dir

        self.reset_state()
        self.name = name
        self.root_dir = root_dir
        self.partial_data = partial_data

        if shuffle is None:
            shuffle = name == 'train'
        self.shuffle = shuffle
        self.image_list = []
        self.gta_list = []
        if name == 'train':
            f = open(os.path.join(meta_dir,"train.txt"),"r")
            with open(gtav_train_list,"r") as f_gta5:
                for line in f_gta5.readlines():
                    filename = line.strip()
                    self.gta_list.append([os.path.join(root_gtav,"images",filename), os.path.join(root_gtav,"labels",filename), 1.0])

            assert self.partial_data == -1
        elif name=="val":
            #assert self.shuffle==False
            f = open(os.path.join(meta_dir, "val.txt"), "r")
        elif name=="test":
            #assert self.shuffle == False
            f = open(os.path.join(meta_dir, "test.txt"), "r")
        else:
            raise ValueError

        for line in f.readlines():
            tmp_list = line.strip("\n").split(" ")
            tmp_list = [os.path.join(self.root_dir,tmp) for tmp in tmp_list]
            self.image_list.append([tmp_list[0], tmp_list[1], 0.0])

        f.close()

        if name == "train":
            self.image_list.extend(self.gta_list[:len(self.image_list)])

        self.image_list = self.image_list[:self.partial_data]

    def size(self):
        return len(self.image_list)

    @staticmethod
    def class_num():
        return 19

    def get_data(self):
        idxs = np.arange(len(self.image_list))
        if self.shuffle:
            self.rng.shuffle(idxs)
        for k in idxs:
            fname, flabel, flag  = self.image_list[k]
            if flag == 0: #cityscapes
                fname = cv2.imread(fname, cv2.IMREAD_COLOR)
                flabel = cv2.imread(flabel, cv2.IMREAD_GRAYSCALE)
            else:
                fname = cv2.imread(fname, cv2.IMREAD_COLOR)
                flabel_origin = np.asarray(Image.open(flabel))

                flabel = 255 * np.ones(flabel_origin.shape, dtype=np.uint8)
                for k, v in id_to_trainid.items():
                    flabel[flabel_origin == k] = v

            yield [fname, flabel, flag]




if __name__ == '__main__':
    dataset = Cityscapes_GTAV(root_dir="data/cityscapes",meta_dir="metadata/cityscapes",name="val",shuffle=True)

    for data in dataset.get_data():
        img = data[0]
        label = data[1]
        flag = data[2]
        from tensorpack.utils.segmentation.segmentation import visualize_label
        cv2.imwrite("img.jpg", np.concatenate((img, visualize_label(label)), axis=1))
        print np.unique(label)
        print flag
        pass

