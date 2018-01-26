# Author: Tao Hu <taohu620@gmail.com>


import os
import gzip
import numpy as np
import cv2
import random
from tensorpack.utils import logger
from tensorpack.dataflow.base import RNGDataFlow
import json
from tensorpack.utils.skeleton import visualization
__all__ = ['mpii']

nr_skeleton = 16 #??
nr_aug_copies =4
pixel_means = np.array([[[102.9801, 115.9465, 122.7717]]]) # BGR
data_shape = (256,256)
output_shape = (64,64)
img_path = "/data1/dataset/mpii/images"

class mpii(RNGDataFlow):
    def __init__(self, img_dir, meta_dir, name,
                 shuffle=None):

        assert name in ['train', 'val'], name

        self.reset_state()
        self.meta_dir = meta_dir
        self.name = name
        self.img_dir = img_dir

        with open(meta_dir) as anno_file:
            ann_list = json.load(anno_file)

        if shuffle is None:
            shuffle = name == 'train'
        self.shuffle = shuffle
        self.imglist = []

        for ann in ann_list:
            if name == "train":
                if ann['isValidation'] == 0:
                    self.imglist.append(ann)
            elif name == "val":
                if ann['isValidation'] == 1:
                    self.imglist.append(ann)
            else:
                raise

        #self.imglist = self.imglist[:20]


    def size(self):
        return len(self.imglist)

    @staticmethod
    def joint_num():
        return 16

    def get_data(self):
        idxs = np.arange(len(self.imglist))
        if self.shuffle:
            self.rng.shuffle(idxs)
        for k in idxs:
            cur = self.imglist[k]
            annolist_index = cur["annolist_index"]
            img_paths = cur["img_paths"]
            joint_self = cur["joint_self"]
            numOtherPeople = cur['numOtherPeople']
            people_index = cur['people_index']
            img_height = cur['img_height']
            img_width = cur['img_width']
            img = cv2.imread(os.path.join(self.img_dir, img_paths))

            heatmaps = joints_heatmap_gen(imgs, labels, output_shape)
            imgs = np.transpose(imgs, (0, 2, 3, 1))
            heatmaps = np.transpose(heatmaps, (0, 2, 3, 1))

            #yield imgs.astype(np.float32), heatmaps.astype(np.float32), metadatas
            yield [img, heatmap, metadatas]



def joints_heatmap_gen(data, label, tar_size, ori_size=data_shape, points=nr_skeleton):
    # generate the skeleton label maps based on the 2d skeleton joint positions

    ret = np.zeros((len(data), points, tar_size[0], tar_size[1]), dtype='float32')
    for i in range(len(ret)):
        for j in range(points):
            if label[i][j << 1] < 0 or label[i][j << 1 | 1] < 0:
                continue
            label[i][j << 1 | 1] = min(label[i][j << 1 | 1], ori_size[0] - 1)
            label[i][j << 1] = min(label[i][j << 1], ori_size[1] - 1)
            ret[i][j][int(label[i][j << 1 | 1] * tar_size[0] / ori_size[0])][
                int(label[i][j << 1] * tar_size[1] / ori_size[1])] = 1
    for i in range(len(ret)):
        for j in range(points):
            ret[i, j] = cv2.GaussianBlur(ret[i, j], (7, 7), 0)
    for i in range(len(ret)):
        for j in range(nr_skeleton):
            am = np.amax(ret[i][j])
            if am == 0:
                continue
            ret[i][j] /= am / 255

    return ret




if __name__ == '__main__':
    mm = mpii('/data1/dataset/mpii/images','/home/hutao/lab/tensorpack-forpr/examples/Hourglass/metadata/mpii_annotations.json',"train")
    for i in mm.get_data():
        pass