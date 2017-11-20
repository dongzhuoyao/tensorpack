#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Tao Hu <taohu620@gmail.com>

from __future__ import print_function
import cv2
import tensorflow as tf
import numpy as np
import os
import argparse

from tensorpack import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *

def tower_func(image):
    with argscope(Conv2D, kernel_shape=3, nl=tf.nn.relu):
        logits = (LinearWrap(image)
                  .Conv2D('conv1_1', 64)
                  .Conv2D('conv1_2', 64)
                  .MaxPooling('pool1', 2)
                  # 112
                  .Conv2D('conv2_1', 128)
                  .Conv2D('conv2_2', 128)
                  .MaxPooling('pool2', 2)
                  # 56
                  .Conv2D('conv3_1', 256)
                  .Conv2D('conv3_2', 256)
                  .Conv2D('conv3_3', 256)
                  .MaxPooling('pool3', 2)
                  # 28
                  .Conv2D('conv4_1', 512)
                  .Conv2D('conv4_2', 512)
                  .Conv2D('conv4_3', 512)
                  .MaxPooling('pool4', 2)
                  # 14
                  .Conv2D('conv5_1', 512)
                  .Conv2D('conv5_2', 512)
                  .Conv2D('conv5_3', 512)
                  .MaxPooling('pool5', 2)
                  # 7
                  .FullyConnected('fc6', 4096, nl=tf.nn.relu)
                  .Dropout('drop0', 0.5)
                  .FullyConnected('fc7', 4096, nl=tf.nn.relu)
                  .Dropout('drop1', 0.5)
                  .FullyConnected('fc8', out_dim=1000, nl=tf.identity)())
    tf.nn.softmax(logits, name='prob')
