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
import scipy
__all__ = ['mpii']

nr_skeleton = 16 #??
nr_aug_copies =4
pixel_means = np.array([[[102.9801, 115.9465, 122.7717]]]) # BGR
data_shape = (256,256)
output_shape = (64,64)
label_type = "Gaussian"
sigma = 1
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

            #TODO Adjust center/scale slightly to avoid cropping limbs

            # For single-person pose estimation with a centered/scaled figure
            c = cur['objpos']
            s = cur['scale_provided']
            inp = crop(img, c, s, [data_shape[0], data_shape[1]], rot=0)
            #TODO color_normalize

            # Generate ground truth
            tpts = np.copy(joint_self)
            target = np.zeros((nr_skeleton, output_shape[0], output_shape[1]))
            for i in range(nr_skeleton):
                # if tpts[i, 2] > 0: # This is evil!!
                if tpts[i, 0] > 0:
                    tpts[i, 0:2] = transform(tpts[i, 0:2] + 1, c, s, [output_shape[0], output_shape[1]], rot=0)
                    target[i] = draw_labelmap(target[i], tpts[i] - 1, sigma, type=label_type)

            # Meta info
            meta = {'index': annolist_index, 'center': c, 'scale': s,
                    'pts': joint_self, 'tpts': tpts}

            target = np.transpose(target, (1, 2, 0))
            yield [inp, target, meta]


def draw_labelmap(img, pt, sigma, type='Gaussian'):
    # Draw a 2D gaussian
    # Adopted from https://github.com/anewell/pose-hg-train/blob/master/src/pypose/draw.py

    # Check that any part of the gaussian is in-bounds
    ul = [int(pt[0] - 3 * sigma), int(pt[1] - 3 * sigma)]
    br = [int(pt[0] + 3 * sigma + 1), int(pt[1] + 3 * sigma + 1)]
    if (ul[0] >= img.shape[1] or ul[1] >= img.shape[0] or
            br[0] < 0 or br[1] < 0):
        # If not, just return the image as is
        return img

    # Generate gaussian
    size = 6 * sigma + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    # The gaussian is not normalized, we want the center value to equal 1
    if type == 'Gaussian':
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    elif type == 'Cauchy':
        g = sigma / (((x - x0) ** 2 + (y - y0) ** 2 + sigma ** 2) ** 1.5)


    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    return img


def get_transform(center, scale, res, rot=0):
    """
    General image processing functions
    """
    # Generate transformation matrix
    h = 200 * scale
    t = np.zeros((3, 3))
    t[0, 0] = float(res[1]) / h
    t[1, 1] = float(res[0]) / h
    t[0, 2] = res[1] * (-float(center[0]) / h + .5)
    t[1, 2] = res[0] * (-float(center[1]) / h + .5)
    t[2, 2] = 1
    if not rot == 0:
        rot = -rot # To match direction of rotation from cropping
        rot_mat = np.zeros((3,3))
        rot_rad = rot * np.pi / 180
        sn,cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0,:2] = [cs, -sn]
        rot_mat[1,:2] = [sn, cs]
        rot_mat[2,2] = 1
        # Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0,2] = -res[1]/2
        t_mat[1,2] = -res[0]/2
        t_inv = t_mat.copy()
        t_inv[:2,2] *= -1
        t = np.dot(t_inv,np.dot(rot_mat,np.dot(t_mat,t)))
    return t


def transform(pt, center, scale, res, invert=0, rot=0):
    # Transform pixel location to different reference
    t = get_transform(center, scale, res, rot=rot)
    if invert:
        t = np.linalg.inv(t)
    new_pt = np.array([pt[0] - 1, pt[1] - 1, 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2].astype(int) + 1

def crop(img, center, scale, res, rot=0):
    # Preprocessing for efficient cropping
    ht, wd = img.shape[0], img.shape[1]
    sf = scale * 200.0 / res[0]
    if sf < 2:
        sf = 1
    else:
        new_size = int(np.math.floor(max(ht, wd) / sf))
        new_ht = int(np.math.floor(ht / sf))
        new_wd = int(np.math.floor(wd / sf))
        if new_size < 2:
            logger.info("here TODO..")
            return #TODO
            #return torch.zeros(res[0], res[1], img.shape[2]) \
            #            if len(img.shape) > 2 else torch.zeros(res[0], res[1])
        else:
            img = scipy.misc.imresize(img, [new_ht, new_wd])
            center = [i * 1.0 / sf for i in center]
            scale = scale / sf

    # Upper left point
    ul = np.array(transform([0, 0], center, scale, res, invert=1))
    # Bottom right point
    br = np.array(transform(res, center, scale, res, invert=1))

    # Padding so that when rotated proper amount of context is included
    pad = int(np.linalg.norm(br - ul) / 2 - float(br[1] - ul[1]) / 2)
    if not rot == 0:
        ul -= pad
        br += pad

    new_shape = [br[1] - ul[1], br[0] - ul[0]]
    if len(img.shape) > 2:
        new_shape += [img.shape[2]]
    new_img = np.zeros(new_shape)

    # Range to fill new array
    new_x = max(0, -ul[0]), min(br[0], len(img[0])) - ul[0]
    new_y = max(0, -ul[1]), min(br[1], len(img)) - ul[1]
    # Range to sample from original image
    old_x = max(0, ul[0]), min(len(img[0]), br[0])
    old_y = max(0, ul[1]), min(len(img), br[1])
    new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]] = img[old_y[0]:old_y[1], old_x[0]:old_x[1]]

    if not rot == 0:
        # Remove padding
        new_img = scipy.misc.imrotate(new_img, rot)
        new_img = new_img[pad:-pad, pad:-pad]

    new_img = scipy.misc.imresize(new_img, res)
    return new_img

if __name__ == '__main__':
    mm = mpii('/data1/dataset/mpii/images','/home/hutao/lab/tensorpack-forpr/examples/Hourglass/metadata/mpii_annotations.json',"train")
    for data in mm.get_data():
        img = data[0]
        feat = data[1]
        meta = data[2]


        feat = np.sum(feat,axis=2)
        cv2.imshow("img", img)
        cv2.imshow("featmap",cv2.resize(feat,(data_shape[0],data_shape[1])))
        cv2.waitKey()
