# Coder: Wenxin Xu
# Github: https://github.com/wenxinxu/resnet_in_tensorflow
# ==============================================================================
'''
This is the resnet structure
'''
import numpy as np
import tensorflow as tf
slim = tf.contrib.slim
from resnet_v1 import bottleneck_hourglass
from resnet_utils import conv2d_same, resnet_arg_scope

BN_EPSILON = 0.001
weight_decay = 1e-4
nr_skeleton = 16

def create_bottleneck(prefix, f_in, stride, depth_bottleneck, depth, has_proj=False):
        with tf.variable_scope(prefix, [f_in]) as sc:
            return bottleneck_hourglass(f_in,depth,depth_bottleneck,stride)


def add_init(module, module_name):
    input_name = 'input'
    with slim.arg_scope([slim.conv2d],
                        activation_fn=None, normalizer_fn=None):
        f = conv2d_same(module[input_name], 64, 7, stride=1, scope=module_name + '_1_convbnrelu')
    f = tf.nn.max_pool(f, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='SAME', name = module_name + '_1_pool')
    f = create_bottleneck(module_name + '_bottleneck1', f, 1, 64, 128, has_proj=True)

    f = tf.nn.max_pool(f, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='SAME', name=module_name + '_2')
    f = create_bottleneck(module_name + '_bottleneck2', f, 1, 64, 128, has_proj=False)
    f = create_bottleneck(module_name + '_bottleneck3', f, 1, 128, 256, has_proj=True)

    module[module_name] = f


def add_hourglass(module, module_name, n, num_channels, input_name) :  #n_max = 4, f_max = 256
    this_name = module_name + '_' + str(n)
    middle_channels = int(num_channels / 2)
    up1 = create_bottleneck(this_name + '_up1', module[input_name], 1, middle_channels, num_channels, has_proj=False)

    f = tf.nn.max_pool(module[input_name], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='SAME', name=this_name+'_p')
    module[this_name+'_low1'] = create_bottleneck(this_name+'_low1', f, 1, middle_channels, num_channels, has_proj=False)

    if n > 1 :
        add_hourglass(module, module_name, n-1, num_channels, this_name+'_low1')
        low2 = module[module_name+'_'+str(n-1)]
    else :
        module[this_name+'_low2'] = create_bottleneck(this_name+'_low2', f, 1, middle_channels, num_channels, has_proj=False)
        low2 = module[this_name+'_low2']


    low3 = create_bottleneck(this_name+'_low3', low2, 1, middle_channels, num_channels, has_proj=False)
    #up2 = tf.image.resize_images(low3, low3.shape[1:3]*tf.constant(2, shape=[2]))
    up2 = tf.image.resize_images(low3, [tf.cast(low3.shape[1],tf.int32)*2,tf.cast(low3.shape[2],tf.int32)*2])
    module[this_name] = up1 + up2
    if n == 4 :
        module[module_name] = module[this_name]
    return module[this_name]

def make_network(data,label, stage, is_training):
    with slim.arg_scope(resnet_arg_scope()):
        module = {}
        output = []
        L = {}
        module['input'] = data
        add_init(module, 'init_1')
        last_input = 'init_1'
        for i in range(stage) :
            f = add_hourglass(module, 'hg_'+str(i)+'_ori', 4, 256, last_input)
            f = create_bottleneck('hg_'+str(i)+'_1', f, 1, 128, 256, has_proj=False)
            f = slim.conv2d(f, 256, [1, 1], stride=1,scope='hg_'+str(i)+'_2_convbnrelu')
            module['hg_'+str(i)+'_2'] = f
            #module['output_' + str(i)] =create_conv(f, input_nums=f.get_shape().as_list()[-1], output_nums=nr_skeleton, stride=1, ksize=1, name='output_'+str(i))
            module['output_' + str(i)] = slim.conv2d(f, nr_skeleton, [1, 1], stride=1,
                                                        scope='output_'+str(i), activation_fn=None,
                                                        normalizer_fn=None)
            if is_training:
                tmploss = tf.losses.mean_squared_error(label, module['output_'+str(i)])
                L["mse{}".format(i)] = tmploss
            output.append(module['output_'+str(i)])

            if i < stage - 1 :
                #module['hg_' + str(i) + '_3'] = create_conv(module['hg_'+str(i)+'_2'], input_nums=module['hg_'+str(i)+'_2'].get_shape().as_list()[-1], output_nums=256, stride=1, ksize=1,
                #            name='hg_'+str(i)+'_3')
                module['hg_' + str(i) + '_3'] = slim.conv2d(module['hg_'+str(i)+'_2'], 256, [1, 1], stride=1,scope='hg_'+str(i)+'_3',activation_fn=None, normalizer_fn=None)
                #module['hg_' + str(i) + '_4'] =create_conv(module['output_'+str(i)],
                #            input_nums=module['output_'+str(i)].get_shape().as_list()[-1], output_nums=256, stride=1,
                 #           ksize=1,name='hg_'+str(i)+'_4')
                module['hg_' + str(i) + '_4'] = slim.conv2d(module['output_'+str(i)], 256, [1, 1], stride=1,
                                                            scope='hg_'+str(i)+'_4', activation_fn=None,
                                                            normalizer_fn=None)

                module['hg_'+str(i)] = module['hg_'+str(i)+'_3'] + module['hg_'+str(i)+'_4']
                module['hg_'+str(i)] = module['hg_'+str(i)] + module[last_input]
                last_input = 'hg_'+str(i)

    return output,L

