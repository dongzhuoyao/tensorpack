# Coder: Wenxin Xu
# Github: https://github.com/wenxinxu/resnet_in_tensorflow
# ==============================================================================
'''
This is the resnet structure
'''
import numpy as np
import tensorflow as tf


BN_EPSILON = 0.001
weight_decay = 1e-4
nr_skeleton = 16



def create_variables(name, shape, initializer=tf.contrib.layers.xavier_initializer(), is_fc_layer=False):
    '''
    :param name: A string. The name of the new variable
    :param shape: A list of dimensions
    :param initializer: User Xavier as default.
    :param is_fc_layer: Want to create fc layer variable? May use different weight_decay for fc
    layers.
    :return: The created variable
    '''

    ## TODO: to allow different weight decay to fully connected layer and conv layer
    if is_fc_layer is True:
        regularizer = tf.contrib.layers.l2_regularizer(scale=weight_decay)
    else:
        regularizer = tf.contrib.layers.l2_regularizer(scale=weight_decay)

    new_variables = tf.get_variable(name, shape=shape, initializer=initializer,
                                    regularizer=regularizer)
    return new_variables




def batch_normalization_layer(input_layer, dimension,name):
    '''
    Helper function to do batch normalziation
    :param input_layer: 4D tensor
    :param dimension: input_layer.get_shape().as_list()[-1]. The depth of the 4D tensor
    :return: the 4D tensor after being normalized
    '''

    mean, variance = tf.nn.moments(input_layer, axes=[0, 1, 2])
    beta = tf.get_variable(name+'-beta', dimension, tf.float32,
                           initializer=tf.constant_initializer(0.0, tf.float32))
    gamma = tf.get_variable(name+'-gamma', dimension, tf.float32,
                            initializer=tf.constant_initializer(1.0, tf.float32))
    bn_layer = tf.nn.batch_normalization(input_layer, mean, variance, beta, gamma, BN_EPSILON,name=name)

    return bn_layer


def conv_bn_relu_layer(prefix,input_layer,ksize,stride,num_outputs,has_bn=True, has_relu=True,conv_name_fun=None, bn_name_fun=None):
    num_iutputs = input_layer.get_shape().as_list()[-1]
    filter_shape = [ksize,ksize,num_iutputs,num_outputs]
    conv_name = prefix
    if conv_name_fun:
        conv_name = conv_name_fun(prefix)
    filter = create_variables(name=conv_name,shape=filter_shape)
    conv_layer = tf.nn.conv2d(input_layer, filter, strides=[1, stride, stride, 1], padding='SAME')
    if has_bn:
        bn_name = "bn_" + prefix
        if bn_name_fun:
            bn_name = bn_name_fun(prefix)
        bn_layer = batch_normalization_layer(conv_layer, num_outputs,name=bn_name)
    else:
        bn_layer = conv_layer
    if has_relu:
        output = tf.nn.relu(bn_layer)
    else:
        output = bn_layer

    return output


def create_bottleneck(prefix, f_in, stride, num_outputs1, num_outputs2, has_proj=False):
    proj = f_in
    if has_proj:
        proj = conv_bn_relu_layer(prefix, f_in, ksize=1, stride=stride, num_outputs=num_outputs2,
            has_bn=True, has_relu=False,
            conv_name_fun=lambda p: "interstellar{}_branch1".format(p),
            bn_name_fun=lambda p: "bn{}_branch1".format(p))

    f = conv_bn_relu_layer(prefix, f_in, ksize=1, stride=stride, num_outputs=num_outputs1,
            has_bn=True, has_relu=True,
            conv_name_fun=lambda p: "interstellar{}_branch2a".format(p),
            bn_name_fun=lambda p: "bn{}_branch2a".format(p))

    f = conv_bn_relu_layer(prefix, f, ksize=3, stride=1, num_outputs=num_outputs1,
            has_bn=True, has_relu=True,
            conv_name_fun=lambda p: "interstellar{}_branch2b".format(p),
            bn_name_fun=lambda p: "bn{}_branch2b".format(p))

    f = conv_bn_relu_layer(prefix, f, ksize=1, stride=1, num_outputs=num_outputs2,
            has_bn=True, has_relu=False,
            conv_name_fun=lambda p: "interstellar{}_branch2c".format(p),
            bn_name_fun=lambda p: "bn{}_branch2c".format(p))

    f = f + proj

    return tf.nn.relu(f)


def add_init(module, module_name):
    input_name = 'input'
    f = conv_bn_relu_layer(module_name + '_1_convbnrelu', module[input_name], ksize=7, stride=1, num_outputs=64)

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
        #low2 = module_name + '_' +str(n-1)
        low2 = module[module_name+'_'+str(n-1)]
    else :
        module[this_name+'_low2'] = create_bottleneck(this_name+'_low2', f, 1, middle_channels, num_channels, has_proj=False)
        low2 = module[this_name+'_low2']
        #low2 = this_name+'_low2'

    low3 = create_bottleneck(this_name+'_low3', low2, 1, middle_channels, num_channels, has_proj=False)
    #up2 = tf.image.resize_images(low3, low3.shape[1:3]*tf.constant(2, shape=[2]))
    up2 = tf.image.resize_images(low3, [tf.cast(low3.shape[1],tf.int32)*2,tf.cast(low3.shape[2],tf.int32)*2])
    module[this_name] = up1 + up2
    if n == 4 :
        module[module_name] = module[this_name]
    return module[this_name]

def create_conv(input,input_nums,output_nums,stride,ksize,name):
    filter_shape = [ksize, ksize, input_nums,output_nums]
    filter = create_variables(name=name, shape=filter_shape)
    return  tf.nn.conv2d(input, filter, strides=[1, stride, stride, 1], padding='SAME')

def make_network(data,label, stage, is_training):
    module = {}
    output = []
    L = {}
    module['input'] = data
    add_init(module, 'init_1')

    last_input = 'init_1'
    for i in range(stage) :
        f = add_hourglass(module, 'hg_'+str(i)+'_ori', 4, 256, last_input)
        f = create_bottleneck('hg_'+str(i)+'_1', f, 1, 128, 256, has_proj=False)
        f = conv_bn_relu_layer('hg_'+str(i)+'_2_convbnrelu', f, ksize=1, stride=1,num_outputs=256)
        module['hg_'+str(i)+'_2'] = f
        module['output_' + str(i)] =create_conv(f, input_nums=f.get_shape().as_list()[-1], output_nums=nr_skeleton, stride=1, ksize=1, name='output_'+str(i))

        if is_training:
            tmploss = tf.losses.mean_squared_error(label, module['output_'+str(i)])
            L["mse{}".format(i)] = tmploss
        output.append(module['output_'+str(i)])

        if i < stage - 1 :
            module['hg_' + str(i) + '_3'] = create_conv(module['hg_'+str(i)+'_2'], input_nums=module['hg_'+str(i)+'_2'].get_shape().as_list()[-1], output_nums=256, stride=1, ksize=1,
                        name='hg_'+str(i)+'_3')
            module['hg_' + str(i) + '_4'] =create_conv(module['output_'+str(i)],
                        input_nums=module['output_'+str(i)].get_shape().as_list()[-1], output_nums=256, stride=1,
                        ksize=1,name='hg_'+str(i)+'_4')

            module['hg_'+str(i)] = module['hg_'+str(i)+'_3'] + module['hg_'+str(i)+'_4']
            module['hg_'+str(i)] = module['hg_'+str(i)] + module[last_input]
            last_input = 'hg_'+str(i)

    return output,L


def bn_relu_conv_layer(input_layer, filter_shape, stride):
    '''
    A helper function to batch normalize, relu and conv the input layer sequentially
    :param input_layer: 4D tensor
    :param filter_shape: list. [filter_height, filter_width, filter_depth, filter_number]
    :param stride: stride size for conv
    :return: 4D tensor. Y = conv(Relu(batch_normalize(X)))
    '''
    in_channel = input_layer.get_shape().as_list()[-1]

    bn_layer = batch_normalization_layer(input_layer, in_channel)
    relu_layer = tf.nn.relu(bn_layer)

    filter = create_variables(shape=filter_shape)
    conv_layer = tf.nn.conv2d(relu_layer, filter, strides=[1, stride, stride, 1], padding='SAME')
    return conv_layer


def residual_block(input_layer, output_channel, first_block=False):
    '''
    Defines a residual block in ResNet
    :param input_layer: 4D tensor
    :param output_channel: int. return_tensor.get_shape().as_list()[-1] = output_channel
    :param first_block: if this is the first residual block of the whole network
    :return: 4D tensor.
    '''
    input_channel = input_layer.get_shape().as_list()[-1]

    # When it's time to "shrink" the image size, we use stride = 2
    if input_channel * 2 == output_channel:
        increase_dim = True
        stride = 2
    elif input_channel == output_channel:
        increase_dim = False
        stride = 1
    else:
        raise ValueError('Output and input channel does not match in residual blocks!!!')

    # The first conv layer of the first residual block does not need to be normalized and relu-ed.
    with tf.variable_scope('conv1_in_block'):
        if first_block:
            filter = create_variables(name='conv', shape=[3, 3, input_channel, output_channel])
            conv1 = tf.nn.conv2d(input_layer, filter=filter, strides=[1, 1, 1, 1], padding='SAME')
        else:
            conv1 = bn_relu_conv_layer(input_layer, [3, 3, input_channel, output_channel], stride)

    with tf.variable_scope('conv2_in_block'):
        conv2 = bn_relu_conv_layer(conv1, [3, 3, output_channel, output_channel], 1)

    # When the channels of input layer and conv2 does not match, we add zero pads to increase the
    #  depth of input layers
    if increase_dim is True:
        pooled_input = tf.nn.avg_pool(input_layer, ksize=[1, 2, 2, 1],
                                      strides=[1, 2, 2, 1], padding='VALID')
        padded_input = tf.pad(pooled_input, [[0, 0], [0, 0], [0, 0], [input_channel // 2,
                                                                      input_channel // 2]])
    else:
        padded_input = input_layer

    output = conv2 + padded_input
    return output
