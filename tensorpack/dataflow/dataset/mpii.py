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



label_type = "Gaussian"
sigma = 1
nr_skeleton = 16

class mpii(RNGDataFlow):
    def __init__(self, img_dir, meta_dir, name, data_shape, output_shape,
                 shuffle=None):

        assert name in ['train', 'val'], name

        self.reset_state()
        self.data_shape = data_shape
        self.output_shape = output_shape
        self.nr_skeleton = nr_skeleton
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

        #self.imglist = self.imglist[:200]


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
            s = s*200
            joint_self = np.array(joint_self)

            image, label, transform_dict =crop_and_padding(img_path=os.path.join(self.img_dir,img_paths),
                                                           objcenter=c, scale=s, joints=joint_self,
                                                           headRect=1, data_shape = self.data_shape, output_shape= self.output_shape, stage=self.name)

            target = np.zeros((self.nr_skeleton, self.output_shape[0], self.output_shape[1]))
            for i in range(self.nr_skeleton):
                # if tpts[i, 2] > 0: # This is evil!!
                if  label[i,0] < self.output_shape[0] and label[i,1] < self.output_shape[1] \
                        and label[i, 0]>0 and label[i,1] > 0:
                        target[i,int(label[i,1]),int(label[i,0])] = 1 #here, notice the order of opencv


            target = np.transpose(target, (1, 2, 0))
            target = cv2.GaussianBlur(target, (3, 3), 0)

            for i in range(nr_skeleton): # normalize to 1, otherwise the peak value may be 0.25, please notice the cv2.GaussianBlur's result.
                    am = np.amax(target[:,:,i])
                    if am == 0:
                        continue
                    target[:, :, i] /= am / 1  # normalize to 1

            # Meta info
            meta = {'index': annolist_index, 'center': c, 'scale': s, 'transform': transform_dict,"meta":cur}

            if self.name == "train":
                yield [image, target]
            elif self.name == "val":
                yield [image, target, meta]
            else:
                raise NotImplementedError



pixel_means = np.array([[[102.9801, 115.9465, 122.7717]]]) # BGR





def crop_and_padding(img_path, objcenter, scale, joints, headRect, data_shape, output_shape, stage):
    img = cv2.imread(img_path)
    add = max(img.shape[0], img.shape[1])
    big_img = cv2.copyMakeBorder(img, add, add, add, add, borderType=cv2.BORDER_CONSTANT,
                              value=pixel_means.reshape(-1))

    joints_origin = np.copy(joints)
    joints[:, 0] += add
    joints[:, 1] += add
    objcenter[0] += add
    objcenter[1] += add
    ###################################################### here is one cheat
    if stage == 'train':
        ext_ratio = 1.25
    elif stage == 'valid':
        ext_ratio = 1.25
    else:
        ext_ratio = 1.

    delta = int(scale * ext_ratio)//2
    min_x = int(objcenter[0]  - delta)
    max_x = int(objcenter[0]  + delta)
    min_y = int(objcenter[1]  - delta)
    max_y = int(objcenter[1]  + delta)

    joints[:, 0] -=  min_x
    joints[:, 1] -=  min_y

    x_ratio = float(output_shape[0]) / (max_x - min_x)
    y_ratio = float(output_shape[1]) / (max_y - min_y)



    joints[:, 0] *= x_ratio
    joints[:, 1] *= y_ratio

    img = cv2.resize(big_img[min_y:max_y, min_x:max_x, :], (data_shape[0], data_shape[1]))

    ind = joints[:, 2].argsort()
    label = joints[ind, :2]

    transform = {}
    transform['divide_first'] = (x_ratio, y_ratio)
    transform['add_second'] = (min_x - add, min_y - add)


    return img, label, transform



if __name__ == '__main__':
    mm = mpii('/data1/dataset/mpii/images','/home/hutao/lab/tensorpack-forpr/examples/Hourglass/metadata/mpii_annotations.json',
              "val",(256,256),(64,64))
    for data in mm.get_data():
        img = data[0]
        feat = data[1]
        meta = data[2]


        feat = np.sum(feat,axis=2)
        cv2.imshow("img", img)
        cv2.imshow("featmap",cv2.resize(feat*255,(256,256)))
        cv2.waitKey(3000)

