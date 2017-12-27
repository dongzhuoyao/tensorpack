import numpy as np
import tensorflow as tf
from resnet_model import resnet_bottleneck
from tensorpack.models import (
    Conv2D, GlobalAvgPooling, BatchNorm, BNReLU, FullyConnected,
    LinearWrap)
import cv2, os
from random import random
nr_skeleton = 16
data_shape = (256, 256)  # height, width
output_shape = (64, 64)  # height, width

nr_aug_copies = 1+1+2 # origin, flip, rotation

pixel_means = np.array([[[102.9801, 115.9465, 122.7717]]])  # BGR
pixel_norm = True
img_path = ""

def add_init(module, module_name):
    input_name = 'input'
    with tf.variable_scope(module_name):
        f = Conv2D('1_convbnrelu', module[input_name],64,7,stride=1,nl = BNReLU)
        f = tf.nn.max_pool(f, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='SAME', name = '1_pool')
        f = resnet_bottleneck(f, 128, stride=1,name="bottlenetck1")
        f = tf.nn.max_pool(f, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='SAME', name='2_pool')
        f = resnet_bottleneck(f, 128, stride=1,name="bottlenetck2")
        f = resnet_bottleneck(f, 256, stride=1,name="bottlenetck3")
        module[module_name] = f


def add_hourglass(module, module_name, n, num_channels, input_name) :  #n_max = 4, f_max = 256
    this_name = module_name + '_' + str(n)
    middle_channels = int(num_channels / 2)
    up1 = resnet_bottleneck(module[input_name], num_channels, stride=1, name=this_name + '_up1')

    f = tf.nn.max_pool(module[input_name], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='SAME', name=this_name+'_p')
    module[this_name+'_low1'] = resnet_bottleneck(f, num_channels, stride=1, name = this_name+'_low1')

    if n > 1 :
        add_hourglass(module, module_name, n-1, num_channels, this_name+'_low1')
        low2 = module[module_name+'_'+str(n-1)]
    else :
        module[this_name+'_low2'] = resnet_bottleneck( f, num_channels,stride=1,name=this_name+'_low2')
        low2 = module[this_name+'_low2']

    low3 = resnet_bottleneck(low2, num_channels, stride=1, name=this_name+'_low3')
    up2 = tf.image.resize_images(low3, low3.shape[1:3]*tf.constant(2, shape=[2]))
    module[this_name] = up1 + up2
    if n == 4 :
        module[module_name] = module[this_name]
    return module[this_name]

def make_network(data,label, phase):
    big_interval = 4
    is_train = 'train' in phase
    module = {}
    output = []
    L = {}
    module['input'] = data
    add_init(module, 'init_1')

    last_input = 'init_1'
    for i in range(big_interval) :
        f = add_hourglass(module, 'hg_'+str(i)+'_ori', 4, 256, last_input)
        f = resnet_bottleneck( f,256, stride=1,name='hg_'+str(i)+'_1')
        f = Conv2D( f, kernel_shape=1, stride=1,out_channel=256,name= 'hg_'+str(i)+'_2_convbnrelu',nl=BNReLU)
        module['hg_'+str(i)+'_2'] = f
        module['output_' + str(i)] =Conv2D(f, out_channel=nr_skeleton, stride=1, kernel_shape=1, name='output_'+str(i))

        if is_train:
            tmploss = tf.losses.mean_squared_error(label, module['output_'+str(i)])
            L["mse{}".format(i)] = tmploss
        output.append(module['output_'+str(i)])

        if i < big_interval - 1:
            module['hg_' + str(i) + '_3'] = Conv2D(module['hg_'+str(i)+'_2'], out_channel=256, stride=1, kernel_shape=1,
                        name='hg_'+str(i)+'_3')
            module['hg_' + str(i) + '_4'] =Conv2D(module['output_'+str(i)],
                         out_channel=256, stride=1,kernel_shape=1,name='hg_'+str(i)+'_4')

            module['hg_'+str(i)] = module['hg_'+str(i)+'_3'] + module['hg_'+str(i)+'_4']
            module['hg_'+str(i)] = module['hg_'+str(i)] + module[last_input]
            last_input = 'hg_'+str(i)

    if is_train:
        nodecay_loss = 0.
        for loss in L.values():
            nodecay_loss += loss / len(L)
        L['total_loss'] = nodecay_loss


    return output,L



##########dataset

def data_pipeline(ds):
    images, labels = ds


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
        if pixel_norm:
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








