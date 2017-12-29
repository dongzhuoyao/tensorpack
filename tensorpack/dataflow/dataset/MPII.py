# Author: Tao Hu <taohu620@gmail.com>


import os
import gzip
import numpy as np
import cv2

from ...utils import logger
from ..base import RNGDataFlow

__all__ = ['MPII']



import os
import numpy as np
import cv2
import random
debug = False

data_shape = shape = (256, 256) #height, width
output_shape = (64, 64)
nr_aug_copies =4
nr_skeleton = 16
pixel_means = np.array([[[102.9801, 115.9465, 122.7717]]]) # BGR
img_path = "/data1/dataset/mpii/images"


def data_augmentation(trainData, trainLabel):
    # the augmented dataset will be (tremNum+1) times (origin + flip + (tremNum-1) rotated) s as the original one
    tremNum = nr_aug_copies - 1  # one flip + (tremNum-1) for rotated
    gotData = trainData.copy()
    trainData = np.append(trainData, [trainData[0] for i in range(tremNum * len(trainData))], axis=0)
    trainLabel = np.append(trainLabel, [trainLabel[0] for i in range(tremNum * len(trainLabel))], axis=0)
    counter = len(gotData)
    for lab in range(len(gotData)):
        ori_img = gotData[lab].transpose(1, 2, 0)
        annot = trainLabel[lab].copy()
        height, width = ori_img.shape[0], ori_img.shape[1]
        center = (width / 2., height / 2.)
        n = nr_skeleton

        affrat = random.uniform(0.75, 1.25)
        halfl_w = min(width - center[0], (width - center[0]) / 1.25 * affrat)
        halfl_h = min(height - center[1], (height - center[1]) / 1.25 * affrat)
        # img = cv2.resize(ori_img[int(center[0] - halfl_w) : int(center[0] + halfl_w + 1), int(center[1] - halfl_h) : int(center[1] + halfl_h + 1)], (width, height))
        img = cv2.resize(ori_img[int(center[1] - halfl_h): int(center[1] + halfl_h + 1),
                         int(center[0] - halfl_w): int(center[0] + halfl_w + 1)], (width, height))
        for i in range(n):
            annot[i << 1] = (annot[i << 1] - center[0]) / halfl_w * (width - center[0]) + center[0]
            annot[i << 1 | 1] = (annot[i << 1 | 1] - center[1]) / halfl_h * (height - center[1]) + center[1]

        trainData[lab] = img.transpose(2, 0, 1)
        trainLabel[lab] = annot

        # flip augmentation
        newimg = cv2.flip(img, 1)
        cod = [];
        allc = []
        for i in range(n):
            x, y = annot[i << 1], annot[i << 1 | 1]
            if x >= 0:
                x = width - 1 - x
            cod.append((x, y))
        trainData[counter] = newimg.transpose(2, 0, 1)

        # **** the joint index depends on the dataset ****
        cod[0], cod[5] = cod[5], cod[0]
        cod[1], cod[4] = cod[4], cod[1]
        cod[2], cod[3] = cod[3], cod[2]
        cod[10], cod[15] = cod[15], cod[10]
        cod[11], cod[14] = cod[14], cod[11]
        cod[12], cod[13] = cod[13], cod[12]
        for i in range(n):
            allc.append(cod[i][0]);
            allc.append(cod[i][1])
        trainLabel[counter] = np.array(allc)
        counter += 1

        # rotated augmentation
        for times in range(tremNum - 1):
            angle = random.uniform(0, 30)
            if random.randint(0, 1):
                angle *= -1
            rotMat = cv2.getRotationMatrix2D(center, angle, 1.0)
            newimg = cv2.warpAffine(img, rotMat, (width, height))

            allc = []
            for i in range(n):
                x, y = annot[i << 1], annot[i << 1 | 1]
                coor = np.array([x, y])
                if x >= 0 and y >= 0:
                    R = rotMat[:, : 2]
                    W = np.array([rotMat[0][2], rotMat[1][2]])
                    coor = np.dot(R, coor) + W
                allc.append(coor[0]);
                allc.append(coor[1])

            newimg = newimg.transpose(2, 0, 1)
            trainData[counter] = newimg
            trainLabel[counter] = np.array(allc)
            counter += 1

    return trainData, trainLabel


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


def preprocessing(data, shape, return_headRect=False, stage='test',return_patch_for_check = False):
    # loading the dataset, generate the images and 2d joint psositions
    height, width = shape
    imgs = []
    patch_imgs = []
    labels = []
    headRects = []
    metadatas = []
    for d in data:
        metadata = {}
        img = cv2.imread(os.path.join(img_path, d['imgpath']))
        add = max(img.shape[0], img.shape[1])
        bimg = cv2.copyMakeBorder(img, add, add, add, add, borderType=cv2.BORDER_CONSTANT,
                                  value=pixel_means.reshape(-1)) #???
        #ipdb.set_trace()
        objcenter = d['objcenter']
        scale = d['scale']
        joints = d['joints'].copy()
        if return_headRect:
            headRect = d['headRect'].copy()

        joints_save = joints.copy()

        joints = joints_save
        joints[:, :2] += add
        if return_headRect:
            headRect += add

        ###################################################### here is one cheat
        if stage == 'train':
            ext_ratio = 1.25
        elif stage == 'valid':
            ext_ratio = 1.25
        else:
            ext_ratio = 1.
            # ext_ratio = random.uniform(0.75, 1.25)
        delta = int(scale * ext_ratio)
        min_x = objcenter[0] + add - delta // 2
        max_x = objcenter[0] + add + delta // 2
        min_y = objcenter[1] + add - delta // 2
        max_y = objcenter[1] + add + delta // 2

        joints[:, 0] = joints[:, 0] - min_x
        joints[:, 1] = joints[:, 1] - min_y

        x_ratio = float(width) / (max_x - min_x)
        y_ratio = float(height) / (max_y - min_y)

        joints[:, 0] *= x_ratio
        joints[:, 1] *= y_ratio

        metadata["min_x"] = min_x-add
        metadata["min_y"] = min_y-add
        metadata["x_ratio"] = x_ratio
        metadata["y_ratio"] = y_ratio

        num_joints = nr_skeleton
        label = np.zeros((num_joints, 2))
        if return_headRect:
            headRect = np.array([headRect[0] - min_x, headRect[1] - min_y, headRect[2] - min_x, headRect[3] - min_y],
                                np.float32)
            headRect[[0, 2]] *= x_ratio
            headRect[[1, 3]] *= y_ratio
        img = cv2.resize(bimg[min_y:max_y, min_x:max_x, :], (data_shape[1], data_shape[0]))

        if joints.shape[0] < num_joints:#???
            if joints.shape[0] != len(np.unique(joints[:, 2])):
                label[:] = -1
                from IPython import embed;
                embed()
            d = list(set(range(num_joints)) - set(joints[:, 2]))
            pad = np.zeros((len(d), 4)) - 1
            pad[:, 2] = np.array(d, dtype=int).reshape(1, -1)
            joints = np.concatenate((joints, pad))

        ind = joints[:, 2].argsort()
        label = joints[ind, :2]

        if  return_patch_for_check:
            from visualization import draw_skeleton_new
            patch_img = draw_skeleton_new(img, label.astype(int))
            patch_img = patch_img.transpose(2, 0, 1)
            patch_imgs.append(patch_img)

        if False:
            from visualization import draw_skeleton_new
            img = draw_skeleton_new(img, label.astype(int))
            print(label)
            print("scale: {}".format(scale))
            # cv2.rectangle(img, (headRect[0],headRect[1]), (headRect[2], headRect[3]), (0, 255,0), 3)
            cv2.imshow('', img)
            cv2.waitKey()

        img = img - pixel_means
        if config.pixel_norm:
            img = img / 255.
        img = img.transpose(2, 0, 1)
        imgs.append(img)
        labels.append(label.reshape(-1))
        if return_headRect:
            headRects.append(headRect)
        metadatas.append(metadata)

    if return_patch_for_check:
        return np.array(patch_imgs), np.array(labels), metadatas

    if return_headRect:
        return np.array(imgs), np.array(labels),np.array(headRects),metadatas
    else:
        return np.array(imgs), np.array(labels),metadatas

class MPII(RNGDataFlow):
    def __init__(self, dir, meta_dir, name,
                 shuffle=None, dir_structure=None):

        assert name in ['train', 'val'], name
        assert os.path.isdir(dir), dir
        self.reset_state()
        self.dir = dir
        self.name = name

        if shuffle is None:
            shuffle = name == 'train'
        self.shuffle = shuffle
        self.imglist = []

        if name == 'train':
            f = open(os.path.join(meta_dir,"train_aug.txt"),"r")
        else:
            f = open(os.path.join(meta_dir, "val.txt"), "r")

        for line in f.readlines():
            self.imglist.append(line.strip("\n").split(" "))
        f.close()

        #self.imglist = self.imglist[:40]

    def size(self):
        return len(self.imglist)

    def get_data(self):
        idxs = np.arange(len(self.imglist))
        if self.shuffle:
            self.rng.shuffle(idxs)
        for k in idxs:
            batchdata = self.imglist[k]
            imgs, labels, metadatas = preprocessing(batchdata, shape, stage=self.name)
            if self.name=="train":
                imgs, labels = data_augmentation(imgs, labels)
            heatmaps = joints_heatmap_gen(imgs, labels, output_shape)
            imgs = np.transpose(imgs, (0, 2, 3, 1))
            heatmaps = np.transpose(heatmaps, (0, 2, 3, 1))




import os
import os.path as osp
import numpy as np
import cv2

'''
skeleton index (person centric):
0: right foot
1: right knee
2: right buttocks
3: left buttocks
4: left knee
5: left foot
6: torso down
7: torso middle
8: neck (torso up)
9: head
10: right hand
11: right elbow
12: right shoulder
13: left shoulder
14: left elbow
15: left hand
'''

cur_dir = os.path.dirname(__file__)

class MPIJoints(object):
    def __init__(self):
        self.max_num_joints = 16
        self.color = np.random.randint(0, 256, (self.max_num_joints, 3))
        self.train_data_path = osp.join(cur_dir, 'GT', 'train_data.pkl')
        self.test_data_path = osp.join(cur_dir, 'GT', 'test_data.pkl')

        self.realtest_data_path = osp.join(cur_dir, 'GT', 'real_test_data.pkl')
        if True:
            import pickle
            with open(self.train_data_path, 'rb') as f:
                self.mpi = pickle.load(f)
            with open(self.test_data_path, 'rb') as f:
                self.test_mpi = pickle.load(f)

            with open(self.realtest_data_path, 'rb') as f:
                self.realtest_mpi = pickle.load(f)
        else:

            gt_path=os.path.join(cur_dir, 'GT', 'single_person_trainInfo.res')
            self.mpi = []
            self.test_mpi = []
            with open( gt_path ) as f:
                lines = f.readlines()


            import json
            with open('../data/GT/test_images.txt', 'r') as f:
                imgnames = json.load(f)
            imgnames = [img.split('/')[-1] for img in imgnames]
            with open('../data/GT/test_headRect.txt', 'r') as f:
                headRect = json.load(f)
            joints_tmp = np.load('./joints.npy')
            headRect = np.array(headRect)

            color = np.random.randint( 0, 255, (16, 3) )
            lineID = 0
            numLines = len( lines )
            while lineID < numLines:
                tmp = lines[lineID].strip().split( ' ' )
                lineID += 1
                imgname = tmp[0]

                if len(self.mpi) % 1000 == 0:
                    print(len(self.mpi))

                numGTs = 1
                for i in range(numGTs):
                    rect = np.array( tmp[1:5], np.int )
                    scale = int(float(tmp[5]) * 200)
                    objcenter = int(tmp[6]), int(tmp[7])

                    numJoints = 0
                    curJoints = []
                    for j in range(16):
                        if j+lineID == len(lines)  or  len(lines[j+lineID].strip().split(' ')) > 3 :
                            break
                        numJoints += 1
                        curJoints.append( lines[j+lineID].strip().split(' ')+[1] ) #add "1" in the end
                    curJoints = np.array(curJoints, np.float)
                    lineID += numJoints

                    if numJoints != len(np.unique(curJoints[:, 2])): continue
                    if numJoints < 4: continue

                    isTest = False
                    for k in range(len(imgnames)):
                        if imgname == imgnames[k] and np.sum(rect - headRect[k]) < 1e-5:
                            if len(curJoints) == len(joints_tmp[k]) and np.sum(np.array(curJoints, np.float)[:, :3] - joints_tmp[k][:, :3]) < 1e-3:
                                isTest = True
                                break

                    joints = np.array( curJoints, np.float )
                    humanData = dict(joints=joints, imgpath=imgname, headRect=rect, scale=scale, objcenter=objcenter)
                    if not isTest:
                        self.mpi.append(humanData)
                    else:
                        self.test_mpi.append(humanData)

    def load_data(self, num_test=2000):
        return self.mpi, self.test_mpi[:num_test]

    def load_realtest_data(self):
        return self.realtest_mpi

    def dump_data(self):
        import pickle
        with open(self.train_data_path, 'wb') as f:
            pickle.dump(self.mpi, f)
        with open(self.test_data_path, 'wb') as f:
            pickle.dump(self.test_mpi, f)

    def evaluate_error(self, test_label, test_res):
        # shape is: num_samples * (num_joints*2)
        assert( test_label.shape == test_res.shape )
        num_samples = test_label.shape[0]
        num_joints = test_label.shape[1]//2
        diff = (test_res-test_label) ** 2
        test_loss = 0
        test_count = 0
        for n in range(num_samples):
            for j in range(num_joints):
                if test_label[n, 2*j]>0 and test_label[n, 2*j+1]:
                    test_loss += np.sqrt(diff[n, 2*j]+diff[n, 2*j+1])
                    test_count += 1
        test_loss /= (test_count)
        print(test_loss)
        return test_loss

    def evaluate_PCK(self, test_label, test_res, w=100, h=200):
        # shape is: num_samples * (num_joints*2)
        assert( test_label.shape == test_res.shape )
        num_samples = test_label.shape[0]
        num_joints = test_label.shape[1]//2
        diff = (test_res-test_label) ** 2
        joint_diff = np.zeros( (num_samples, num_joints) )
        test_available = test_label[:, :-1:2] > 0
        for j in range(num_joints):
            joint_diff[:, j] = np.sqrt(diff[:, 2*j]+diff[:, 2*j+1])
        num_ratios = 3
        pck_res = np.zeros( (num_ratios, num_joints) )
        result = []
        for i, ratio in enumerate([ 0.05, 0.1, 0.2 ]):
            thresh = ratio * max(w, h)
            res_joints = np.zeros((num_joints))
            res_joints_count = np.zeros((num_joints))
            for n in range(num_samples):
                for j in range(num_joints):
                    if test_available[n, j]:
                        res_joints[j] += (joint_diff[n, j] <= thresh)
                        res_joints_count[j] += 1

            pck_res[i] = res_joints / res_joints_count
            print('PCK@{}: {}'.format(ratio, np.sum(pck_res[i])/float(num_joints) ))
            result.append((ratio,np.sum(pck_res[i])/float(num_joints)))
        #return pck_res      # shape: #ratio * #joints
        return result

    def evaluate_PCKh(self, test_label, test_res, test_rect):
        # shape is: num_samples * (num_joints*2)
        assert( test_label.shape == test_res.shape )
        num_samples = test_label.shape[0]
        num_joints = test_label.shape[1]//2
        diff = (test_res-test_label) ** 2
        joint_diff = np.zeros( (num_samples, num_joints) )
        test_available = (test_label[:, :-1:2] >= 0) * (test_label[:, :-1:2] < 256)#in case some label broken
        for j in range(num_joints):
            joint_diff[:, j] = np.sqrt(diff[:, 2*j]+diff[:, 2*j+1])
        res_joints = np.zeros((num_joints))
        res_joints_count = np.zeros((num_joints))
        rect_size = np.max( np.array([test_rect[:, 3]-test_rect[:, 1], test_rect[:,2]-test_rect[:,0]]), axis=0)
        thresh = 0.5 * np.tile(rect_size, (num_joints, 1)).transpose(1, 0)
        for n in range(num_samples):
            for j in range(num_joints):
                if test_available[n, j]:
                    res_joints[j] += (joint_diff[n, j] <= thresh[n, j])
                    res_joints_count[j] += 1

        pckh_res = res_joints / res_joints_count
        print('PCKh@0.5: {}'.format(np.mean(pckh_res)))
        #return pckh_res      # shape:  #joints
        return np.mean(pckh_res)

    def visualize_image( self, img, rect ):
        aa = np.zeros( img.shape, np.uint8 )
        cv2.rectangle( aa, (rect[0], rect[1]), (rect[2], rect[3]), (0, 0, 255), 3 )
        cv2.imshow( "aa", aa )
        cv2.imshow( "img", img )
        cv2.waitKey()

    def visualize_skeleton(self, img, sk):
        aa = np.zeros( img.shape, dtype=np.uint8 )
        for j in range(self.max_num_joints):
            if sk[j][0] > 0 and sk[j][1] > 0:
                cv2.circle( aa, tuple(sk[j]), 2, tuple(self.color[j]), 2 )
        def draw_line( img, p1, p2, color='y' ):
            if color == 'y':       # right part
                c = (0, 255, 255)
            elif color == 'm':   # left part
                c = (255, 0, 255)
            else:                  # middle torso
                c = (0, 0, 255)
            if p1[0] > 0 and p1[1] > 0 and p2[0] > 0 and p2[1] > 0:
                cv2.line( aa, tuple(p1), tuple(p2), c, 2 )
        draw_line( aa, sk[0], sk[1] )
        draw_line( aa, sk[1], sk[2] )
        draw_line( aa, sk[2], sk[6] )
        draw_line( aa, sk[6], sk[3], 'm' )
        draw_line( aa, sk[3], sk[4], 'm' )
        draw_line( aa, sk[4], sk[5], 'm' )
        draw_line( aa, sk[6], sk[7], 'r' )
        draw_line( aa, sk[7], sk[8], 'r' )
        draw_line( aa, sk[8], sk[9], 'r' )
        draw_line( aa, sk[8], sk[12] )
        draw_line( aa, sk[12], sk[11] )
        draw_line( aa, sk[11], sk[10] )
        draw_line( aa, sk[8], sk[13], 'm' )
        draw_line( aa, sk[13], sk[14], 'm' )
        draw_line( aa, sk[14], sk[15], 'm' )

        cv2.imshow( "res", aa )
        cv2.imshow( "ori", img )
        cv2.waitKey()

if __name__ == '__main__':
    train,test = MPIJoints().load_data()
    print("ok")
