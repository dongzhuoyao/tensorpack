import os
import numpy as np
import cv2
from config import config
import random
debug = False

def data_augmentation(trainData, trainLabel):
    # the augmented dataset will be (tremNum+1) times (origin + flip + (tremNum-1) rotated) s as the original one
    tremNum = config.nr_aug_copies - 1  # one flip + (tremNum-1) for rotated
    gotData = trainData.copy()
    trainData = np.append(trainData, [trainData[0] for i in range(tremNum * len(trainData))], axis=0)
    trainLabel = np.append(trainLabel, [trainLabel[0] for i in range(tremNum * len(trainLabel))], axis=0)
    counter = len(gotData)
    for lab in range(len(gotData)):
        ori_img = gotData[lab].transpose(1, 2, 0)
        annot = trainLabel[lab].copy()
        height, width = ori_img.shape[0], ori_img.shape[1]
        center = (width / 2., height / 2.)
        n = config.nr_skeleton

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


def joints_heatmap_gen(data, label, tar_size, ori_size=config.data_shape, points=config.nr_skeleton):
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
        for j in range(config.nr_skeleton):
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
        img = cv2.imread(os.path.join(config.img_path, d['imgpath']))
        add = max(img.shape[0], img.shape[1])
        bimg = cv2.copyMakeBorder(img, add, add, add, add, borderType=cv2.BORDER_CONSTANT,
                                  value=config.pixel_means.reshape(-1)) #???
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

        num_joints = config.nr_skeleton
        label = np.zeros((num_joints, 2))
        if return_headRect:
            headRect = np.array([headRect[0] - min_x, headRect[1] - min_y, headRect[2] - min_x, headRect[3] - min_y],
                                np.float32)
            headRect[[0, 2]] *= x_ratio
            headRect[[1, 3]] *= y_ratio
        img = cv2.resize(bimg[min_y:max_y, min_x:max_x, :], (config.data_shape[1], config.data_shape[0]))

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

        img = img - config.pixel_means
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


def DataIter(stage, data, shape, return_headRect=False, seed=None):
    if seed:
        np.random.seed(seed)
        print('set random seed', seed)
    AUG = config.data_aug and (stage == 'train')
    height, width = shape
    if stage == 'train':
        print('shuffle data')
        np.random.shuffle(data)
    i = 0
    while True:
        it = i + config.batch_size // (config.nr_aug_copies if AUG else 1)
        if i % len(data) < it % len(data):
            batchdata = data[(i % len(data)):(it % len(data))]
        else:
            batchdata = data[(i % len(data)):]
            batchdata.extend(data[:(it % len(data))])

        imgs, labels, metadatas = preprocessing(batchdata, shape, stage=stage)
        if AUG: imgs, labels = data_augmentation(imgs, labels)
        heatmaps = joints_heatmap_gen(imgs, labels, config.output_shape)
        imgs = np.transpose(imgs, (0, 2, 3, 1))
        heatmaps = np.transpose(heatmaps, (0, 2, 3, 1))
        yield imgs.astype(np.float32), heatmaps.astype(np.float32), metadatas
        i = it

