from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf

import densenet_utils

slim = tf.contrib.slim
dense_arg_scope = densenet_utils.dense_arg_scope


def my_squeeze_excitation_layer(input_x, out_dim, ratio, layer_name):
  with tf.variable_scope(layer_name):
    squeeze = tf.reduce_mean(input_x, [1, 2], name='gap', keep_dims=False)

    with tf.variable_scope('fc1'):
      excitation = tf.layers.dense(inputs=squeeze, use_bias=True, units=int(out_dim / ratio))

    excitation = tf.nn.relu(excitation)

    with tf.variable_scope('fc2'):
      excitation = tf.layers.dense(inputs=excitation, use_bias=True, units=out_dim)
    excitation = tf.nn.sigmoid(excitation)

    excitation = tf.reshape(excitation, [-1, 1, 1, out_dim])
    scale = input_x * excitation
    return scale

@slim.add_arg_scope
def unit(inputs, depth, kernel, stride=1, rate=1, drop=0):
  """Basic unit. BN -> RELU -> CONV
  Args:
    inputs: A tensor of size [batch, height, width, channels].
    depth: The growth rate of the composite function layer.
           The num_outputs of bottleneck and transition layer.
    kernel: Kernel size.
    stride: The DenseNet unit's stride.
    rate: An integer, rate for atrous convolution.
    drop: The dropout rate of the DenseNet unit.
  Returns:
    The basic unit's output.
  """
  net = slim.batch_norm(inputs, activation_fn=tf.nn.relu, scope='preact')
  net = slim.conv2d(net, num_outputs=depth, kernel_size=kernel, 
    stride=stride, rate=rate, scope='conv1')
  if drop > 0:
    net = slim.dropout(net, keep_prob=1-drop, scope='dropout')
  return net

@slim.add_arg_scope
def dense(inputs, growth, bottleneck=True, stride=1, rate=1, drop=0,
          outputs_collections=None, scope=None):
  """Dense layer.
  Args:
    inputs: A tensor of size [batch, height, width, channels].
    growth: The growth rate of the dense layer.
    bottleneck: Whether to use bottleneck.
    stride: The DenseNet unit's stride. Determines the amount of downsampling
    of the units output compared to its input.
    rate: An integer, rate for atrous convolution.
    drop: The dropout rate of the dense layer.
    outputs_collections: Collection to add the dense layer output.
    scope: Optional variable_scope.
  Returns:
    The dense layer's output.
  """
  net = inputs
  if bottleneck:
    with tf.variable_scope('bottleneck', values=[net]):
      net = unit(net, depth=4*growth, kernel=[1,1], stride=stride, 
        rate=rate, drop=drop)
  
  with tf.variable_scope('composite', values=[net]):
    net = unit(net, depth=growth, kernel=[3,3], stride=stride, rate=rate, 
      drop=drop)

  return net

@slim.add_arg_scope
def transition(inputs,transition_senet, remove_latter_pooling, bottleneck=True, compress=0.5, stride=1, rate=1, drop=0,
               outputs_collections=None, scope=None):
  """Transition layer.
  Args:
    inputs: A tensor of size [batch, height, width, channels].
    bottleneck: Whether to use bottleneck.
    compress: The compression ratio of the transition layer.
    stride: The transition layer's stride. Determines the amount of downsampling of the units output compared to its input.
    rate: An integer, rate for atrous convolution.
    drop: The dropout rate of the transition layer.
    outputs_collections: Collection to add the transition layer output.
    scope: Optional variable_scope.
  Returns:
    The transition layer's output.
  """
  if transition_senet == 1:
    net = inputs

    if compress < 1:
      num_outputs = math.floor(inputs.get_shape().as_list()[3] * compress)
    else:
      num_outputs = inputs.get_shape().as_list()[3]

    net = unit(net, depth=num_outputs, kernel=[1,1], stride=1,
          rate=rate)
    if stride > 1:
        net = slim.avg_pool2d(net, kernel_size=[2,2], stride=stride, scope='avg_pool')
    else:
      if not remove_latter_pooling:
        net = slim.avg_pool2d(net, kernel_size=[2, 2], stride=stride, scope='avg_pool')


    net = my_squeeze_excitation_layer(net, net.get_shape().as_list()[-1], transition_senet,
                                        "transition_se")

    if drop > 0:
      net = slim.dropout(net, keep_prob=1-drop, scope='dropout')
    return net

  elif transition_senet == 2:
    net = inputs

    if compress < 1:
      num_outputs = math.floor(inputs.get_shape().as_list()[3] * compress)
    else:
      num_outputs = inputs.get_shape().as_list()[3]
    with tf.variable_scope('unit1'):
      net = unit(net, depth=num_outputs//4, kernel=[1,1], stride=1, rate=rate)
    with tf.variable_scope('unit2'):
      net = unit(net, depth=num_outputs//4, kernel=[3,3], stride=stride, rate=rate)
    with tf.variable_scope('unit3'):
      net = unit(net, depth=num_outputs, kernel=[1,1], stride= 1, rate=rate)

    net = my_squeeze_excitation_layer(net, net.get_shape().as_list()[-1], transition_senet,
                                        "transition_se")

    if stride > 1:
      with tf.variable_scope('unit_skip'):
        inputs = unit(inputs,depth=num_outputs,kernel=[1,1],stride=stride,rate=rate)

    net = net + inputs # skip layer

    if drop > 0:
      net = slim.dropout(net, keep_prob=1-drop, scope='dropout')
    return net

  else:
    net = inputs

    if compress < 1:
      num_outputs = math.floor(inputs.get_shape().as_list()[3] * compress)
    else:
      num_outputs = inputs.get_shape().as_list()[3]

    net = unit(net, depth=num_outputs, kernel=[1, 1], stride=1,
               rate=rate)
    if stride > 1:
      net = slim.avg_pool2d(net, kernel_size=[2, 2], stride=stride, scope='avg_pool')
    else:
      if not remove_latter_pooling:
        net = slim.avg_pool2d(net, kernel_size=[2, 2], stride=stride, scope='avg_pool')

    if drop > 0:
      net = slim.dropout(net, keep_prob=1 - drop, scope='dropout')
    return net


@slim.add_arg_scope
def stack_dense_blocks(inputs,num_classes, blocks, growth, remove_latter_pooling, senet,transition_senet, denseindense, bottleneck=True, compress=0.5,
  stride=11, rate=1, drop=0, outputs_collections=None, scope=None):
  """Dense block.
  Args:
    inputs: A tensor of size [batch, height, width, channels].
    blocks: List of number of layers in each block.
    growth: The growth rate of the dense layer.
    bottleneck: Whether to use bottleneck.
    compress: The compression ratio of the transition layer.
    stride: The dense layer's stride. Determines the amount of downsampling of the units output compared to its input.
    rate: An integer, rate for atrous convolution.
    drop: The dropout rate of the transition layer.
    outputs_collections: Collection to add the dense layer output.
    scope: Optional variable_scope.
  Returns:
    The dense block's output.
  """
  net = inputs
  denseindense_list = []
  for i, num_layer in enumerate(blocks):
    with tf.variable_scope('block%d' %(i+1), [net]) as sc_block:
      for j in range(num_layer):
        with tf.variable_scope('dense%d' %(j+1), values=[net]) as sc_layer:
          identity = tf.identity(net)
          dense_output= dense(net, growth, bottleneck, stride = 1, rate= rate[i], drop = 0) # disable dropout in dense conv;

          if senet > 0:
            output_dim = dense_output.get_shape().as_list()[-1]
            dense_output = my_squeeze_excitation_layer(dense_output,output_dim,ratio=senet,layer_name='seblock{}'.format(j+1))#TODO SENet

          net = tf.concat([identity, dense_output], axis=3,
            name='concat%d' %(j+1))

      net = slim.utils.collect_named_outputs(outputs_collections, 
        sc_block.name, net)

    if i < len(blocks) - 1: # last block doesn't have transition
      denseindense_list.append(net)
      with tf.variable_scope('trans%d' %(i+1), values=[net]) as sc_trans:
        net = transition(net,transition_senet,remove_latter_pooling, bottleneck, compress, stride[i], rate=1, drop=0)# enable dropout in transition;

        net = slim.utils.collect_named_outputs(outputs_collections, 
          sc_trans.name, net)

  if denseindense == 1:
    with tf.variable_scope('denseindense'):
        denseindense_list[0] = smoothen(denseindense_list[0],'0')
        denseindense_list[0] = slim.avg_pool2d(denseindense_list[0], kernel_size=[2, 2], stride=2, scope='avg_pool0')

        denseindense_list[1] = smoothen(denseindense_list[1], '1')

        denseindense_list[2] = smoothen(denseindense_list[2], '2')

        fpn = tf.concat([denseindense_list[0],denseindense_list[1]], axis=3,name='concat')
        fpn = slim.conv2d(fpn, num_outputs=fpn.get_shape().as_list()[-1],
                                 kernel_size=1,stride=1, rate=1, scope='conv1') #smooth
        fpn = slim.avg_pool2d(fpn, kernel_size=[2, 2], stride=2, scope='avg_pool')

        fpn = tf.concat([fpn,denseindense_list[2]],axis=3,name='concat2')
        fpn = slim.conv2d(fpn, num_outputs=fpn.get_shape().as_list()[-1],
                          kernel_size=1, stride=1, rate=1, scope='conv2')  # smooth
        return tf.concat([fpn,net],axis=3,name='concat2')

  elif denseindense == 2:
    with tf.variable_scope('denseindense'):
        denseindense_list[0] = smoothen(denseindense_list[0],'0')
        denseindense_list[0] = slim.avg_pool2d(denseindense_list[0], kernel_size=[2, 2], stride=2, scope='avg_pool0')

        denseindense_list[1] = smoothen(denseindense_list[1], '1')

        denseindense_list[2] = smoothen(denseindense_list[2], '2')

        fpn = tf.concat([denseindense_list[0],denseindense_list[1]], axis=3,name='concat')
        fpn = slim.batch_norm(fpn, activation_fn=tf.nn.relu, scope='preact1')
        fpn = slim.conv2d(fpn, num_outputs=fpn.get_shape().as_list()[-1],
                                 kernel_size=1,stride=1, rate=1, scope='conv1') #smooth
        fpn = slim.avg_pool2d(fpn, kernel_size=[2, 2], stride=2, scope='avg_pool')

        fpn = tf.concat([fpn,denseindense_list[2]],axis=3,name='concat2')

        fpn = slim.batch_norm(fpn, activation_fn=tf.nn.relu, scope='preact2')
        fpn = slim.conv2d(fpn, num_outputs=fpn.get_shape().as_list()[-1],
                          kernel_size=1, stride=1, rate=1, scope='conv2')  # smooth
        fpn = tf.concat([fpn,net],axis=3,name='concat2')

        fpn = my_squeeze_excitation_layer(fpn,fpn.get_shape().as_list()[-1],4,'final_se')
        return fpn

  elif denseindense == 3:
    with tf.variable_scope('denseindense'):
      channel = 800
      denseindense_list[0] = smoothen(denseindense_list[0], '0',output_num=channel)
      denseindense_list[0] = slim.avg_pool2d(denseindense_list[0], kernel_size=[2, 2], stride=2, scope='avg_pool0')

      denseindense_list[1] = smoothen(denseindense_list[1], '1',output_num=channel)

      denseindense_list[2] = smoothen(denseindense_list[2], '2',output_num=channel)

      fpn = denseindense_list[0]+denseindense_list[1]
      fpn = slim.conv2d(fpn, num_outputs=fpn.get_shape().as_list()[-1],
                        kernel_size=1, stride=1, rate=1, scope='conv1')  # smooth
      fpn = slim.avg_pool2d(fpn, kernel_size=[2, 2], stride=2, scope='avg_pool')

      fpn = fpn + denseindense_list[2]

      fpn = slim.conv2d(fpn, num_outputs=fpn.get_shape().as_list()[-1],
                        kernel_size=1, stride=1, rate=1, scope='conv2')  # smooth
      return fpn + net #TODO buggy

  elif denseindense == 4:
      #https://github.com/kuangliu/pytorch-fpn/blob/master/fpn.py
      def _upsamle_add(x,y,upsample = True):
          if upsample:
            return tf.image.resize_bilinear(x, y.shape[1:3]) + y
          else:
              return x+y
      def _smooth(x, output_channel, name):
          return slim.conv2d(x, num_outputs=output_channel, kernel_size=3,
                                             stride=1, rate=1, scope=name)

      with tf.variable_scope('denseindense'):
          output_channel = 416

          net = slim.conv2d(net, num_outputs=output_channel, kernel_size=1,
                                             stride=1, rate=1, scope='net_conv')
          denseindense_list[2] = slim.conv2d(denseindense_list[2], num_outputs=output_channel, kernel_size=1,
                                             stride=1, rate=1, scope='conv2')
          denseindense_list[1] = slim.conv2d(denseindense_list[1], num_outputs=output_channel, kernel_size=1,
                                             stride=1, rate=1, scope='conv1')
          denseindense_list[0] = slim.conv2d(denseindense_list[0], num_outputs=output_channel, kernel_size=1,
                      stride=1, rate=1, scope='conv0')

          fpn = _upsamle_add(net, denseindense_list[2],upsample=False)
          fpn = _smooth(fpn,output_channel,'smooth1')
          fpn = _upsamle_add(fpn,denseindense_list[1],upsample=True)
          fpn = _smooth(fpn, output_channel, 'smooth2')
          fpn = _upsamle_add(fpn, denseindense_list[0], upsample=True)
          #fpn = _smooth(fpn, output_channel, 'smooth3') directly upsample to origin image size, don't need smooth any more
          return fpn
  elif denseindense == 5:
      #https://github.com/kuangliu/pytorch-fpn/blob/master/fpn.py
      def _upsamle_add(x,y,upsample = True):
          if upsample:
            return tf.image.resize_bilinear(x, y.shape[1:3]) + y
          else:
              return x+y
      def _smooth(x, output_channel, name):
          return slim.conv2d(x, num_outputs=output_channel, kernel_size=1,
                                             stride=1, rate=1, scope=name)

      with tf.variable_scope('denseindense'):
          output_channel = 416

          net = slim.conv2d(net, num_outputs=output_channel, kernel_size=1,
                                             stride=1, rate=1, scope='net_conv')
          denseindense_list[2] = slim.conv2d(denseindense_list[2], num_outputs=output_channel, kernel_size=1,
                                             stride=1, rate=1, scope='conv2')
          denseindense_list[1] = slim.conv2d(denseindense_list[1], num_outputs=output_channel, kernel_size=1,
                                             stride=1, rate=1, scope='conv1')
          denseindense_list[0] = slim.conv2d(denseindense_list[0], num_outputs=output_channel, kernel_size=1,
                      stride=1, rate=1, scope='conv0')

          fpn = _upsamle_add(net, denseindense_list[2],upsample=False)
          fpn = _smooth(fpn,output_channel,'smooth1')
          fpn = _upsamle_add(fpn,denseindense_list[1],upsample=True)
          fpn = _smooth(fpn, output_channel, 'smooth2')
          fpn = _upsamle_add(fpn, denseindense_list[0], upsample=True)
          #fpn = _smooth(fpn, output_channel, 'smooth3') directly upsample to origin image size, don't need smooth any more
          return fpn
  elif denseindense == 6:
      #https://github.com/kuangliu/pytorch-fpn/blob/master/fpn.py
      def _upsamle_add(x,y,upsample = True):
          if upsample:
            return tf.image.resize_bilinear(x, y.shape[1:3]) + y
          else:
              return x+y
      def _smooth(x, output_channel, name):
          return slim.conv2d(x, num_outputs=output_channel, kernel_size=1,
                                             stride=1, rate=1, scope=name)

      def _finalconv(x, output_channel, name):
          with tf.variable_scope(name):
              x = slim.batch_norm(x, activation_fn=tf.nn.relu, scope='postnorm')
              x = slim.conv2d(x, num_outputs=output_channel, kernel_size=1,
                                stride=1, rate=6, scope='conv2classnum')  # dilation 2,4,6
              return x

      with tf.variable_scope('denseindense'):
          output_channel = 416
          class_num = num_classes

          net = slim.conv2d(net, num_outputs=output_channel, kernel_size=1,
                                             stride=1, rate=1, scope='net_conv')
          denseindense_list[2] = slim.conv2d(denseindense_list[2], num_outputs=output_channel, kernel_size=1,
                                             stride=1, rate=1, scope='conv2')
          denseindense_list[1] = slim.conv2d(denseindense_list[1], num_outputs=output_channel, kernel_size=1,
                                             stride=1, rate=1, scope='conv1')
          denseindense_list[0] = slim.conv2d(denseindense_list[0], num_outputs=output_channel, kernel_size=1,
                      stride=1, rate=1, scope='conv0')

          fpn_list = []
          fpn = _upsamle_add(net, denseindense_list[2],upsample=False)
          fpn_list.append(_finalconv(fpn,class_num,'ds0'))
          fpn = _smooth(fpn,output_channel,'smooth1')
          fpn = _upsamle_add(fpn,denseindense_list[1],upsample=True)
          fpn_list.append(_finalconv(fpn, class_num, 'ds1'))
          fpn = _smooth(fpn, output_channel, 'smooth2')
          fpn = _upsamle_add(fpn, denseindense_list[0], upsample=True)
          fpn_list.append(_finalconv(fpn, class_num, 'ds2'))
          #fpn = _smooth(fpn, output_channel, 'smooth3') directly upsample to origin image size, don't need smooth any more
          return fpn_list

  elif denseindense == 7:
      #https://github.com/kuangliu/pytorch-fpn/blob/master/fpn.py
      def _upsamle_add(x,y,upsample = True):
          if upsample:
            return tf.image.resize_bilinear(x, y.shape[1:3]) + y
          else:
              return x+y
      def _smooth(x, output_channel, name):
          return slim.conv2d(x, num_outputs=output_channel, kernel_size=1,
                                             stride=1, rate=1, scope=name)

      def _finalconv(x, output_channel, name):
          with tf.variable_scope(name):
              x = slim.batch_norm(x, activation_fn=tf.nn.relu, scope='postnorm')
              x = slim.conv2d(x, num_outputs=output_channel, kernel_size=1,
                                stride=1, rate=6, scope='conv2classnum')  # dilation 2,4,6
              return x

      with tf.variable_scope('denseindense'):
          output_channel = 416
          class_num = num_classes

          net = slim.conv2d(net, num_outputs=output_channel, kernel_size=1,
                                             stride=1, rate=1, scope='net_conv')
          denseindense_list[2] = slim.conv2d(denseindense_list[2], num_outputs=output_channel, kernel_size=1,
                                             stride=1, rate=1, scope='conv2')
          denseindense_list[1] = slim.conv2d(denseindense_list[1], num_outputs=output_channel, kernel_size=1,
                                             stride=1, rate=1, scope='conv1')
          denseindense_list[0] = slim.conv2d(denseindense_list[0], num_outputs=output_channel, kernel_size=1,
                      stride=1, rate=1, scope='conv0')

          net = my_squeeze_excitation_layer(net,
                                            net.get_shape().as_list()[-1], 4,
                                                             "net_conv_se")
          denseindense_list[2] = my_squeeze_excitation_layer(denseindense_list[2],denseindense_list[2].get_shape().as_list()[-1], 4, "conv2_se")
          denseindense_list[1] = my_squeeze_excitation_layer(denseindense_list[1],
                                                             denseindense_list[1].get_shape().as_list()[-1], 4,
                                                             "conv1_se")
          denseindense_list[0] = my_squeeze_excitation_layer(denseindense_list[0],
                                                             denseindense_list[0].get_shape().as_list()[-1], 4,
                                                             "conv0_se")

          fpn_list = []
          fpn = _upsamle_add(net, denseindense_list[2],upsample=False)
          fpn_list.append(_finalconv(fpn,class_num,'ds0'))
          fpn = _smooth(fpn,output_channel,'smooth1')
          fpn = _upsamle_add(fpn,denseindense_list[1],upsample=True)
          fpn_list.append(_finalconv(fpn, class_num, 'ds1'))
          fpn = _smooth(fpn, output_channel, 'smooth2')
          fpn = _upsamle_add(fpn, denseindense_list[0], upsample=True)
          fpn_list.append(_finalconv(fpn, class_num, 'ds2'))
          #fpn = _smooth(fpn, output_channel, 'smooth3') directly upsample to origin image size, don't need smooth any more
          return fpn_list

  elif denseindense == 0:
    return net
  else:
    raise

def smoothen(net,name, dense_ratio = 4,output_num=0):
  with tf.variable_scope(name):
    if output_num == 0:
      output_num = net.get_shape().as_list()[-1] // dense_ratio

    net = slim.batch_norm(net, activation_fn=tf.nn.relu, scope='preact')
    net = slim.conv2d(net, num_outputs=output_num, kernel_size=1,
                      stride=1, rate=1, scope='conv')

    net = my_squeeze_excitation_layer(net, net.get_shape().as_list()[-1], 4, "se")
    return net
def densenet(inputs,
             blocks,
             rate,
             stride,
             weight_decay,
             growth=32,
             bottleneck=True,
             compress=0.5,
             drop=0,
             stem = 0,
             senet = 0,
             transition_senet=0,
             denseindense = 0,
             num_classes=None,
             is_training=True,
             data_name=None,
             remove_latter_pooling = False,
             reuse=None,
             scope=None):
  """Generator for DenseNet models.
  Args:
    inputs: A tensor of size [batch, height_in, width_in, channels].
    blocks: A list of length equal to the number of DenseNet blocks. Each 
    element is a densenet_utils.DenseBlock object describing the units in the 
    block.
    growth: The growth rate of the DenseNet unit.
    bottleneck: Whether to use bottleneck.
    compress: The compression ratio of the transition layer.
    stride: The dense layer's stride. Determines the amount of downsampling of the units output compared to its input.
    drop: The dropout rate of the transition layer.
    num_classes: Number of predicted classes for classification tasks.
      If 0 or None, we return the features before the logit layer.
    is_training: Whether batch_norm and drop_out layers are in training mode.
    data_name: Which type of model to use.
    reuse: whether or not the network and its variables should be reused. To be
      able to reuse 'scope' must be given.
    scope: Optional variable_scope.
  Returns:
    net: A rank-4 tensor of size [batch, height_out, width_out, channels_out].
      If num_classes is 0 or None, then net is the output of the last DenseNet
      block, potentially after global average pooling. If num_classes is a 
      non-zero integer, net contains the pre-softmax activations.
    end_points: A dictionary from components of the network to the 
    corresponding activation.
  """

  assert len(stride) == len(blocks)
  assert len(rate) == len(blocks)

  with tf.variable_scope(scope, 'densenet', [inputs], reuse=reuse) as sc:
    end_points_collection = sc.original_name_scope + '_end_points'
    with slim.arg_scope(dense_arg_scope(weight_decay=weight_decay)):
      with slim.arg_scope([slim.conv2d, slim.batch_norm, stack_dense_blocks],
                          outputs_collections=end_points_collection):
        with slim.arg_scope([slim.batch_norm, slim.dropout], 
          is_training=is_training):
          net = inputs
            
          if data_name is 'imagenet':
            if stem == 0 :
              net = slim.conv2d(net, growth * 2, kernel_size=[7, 7], stride=2,
                                scope='conv1')
              net = slim.max_pool2d(net, [3, 3], padding='SAME', stride=2,
                                    scope='pool1')
            elif stem == 1:
              net = slim.conv2d(net, 64, kernel_size=[3, 3], stride=2,
                                scope='conv1')
              net = slim.conv2d(net, 64, kernel_size=[3, 3], stride=1,
                                scope='conv1_1')
              net = slim.conv2d(net, 128, kernel_size=[3, 3], stride=1,
                                scope='conv1_2')
              net = slim.max_pool2d(net, [2, 2], padding='SAME', stride=2,
                                    scope='pool1')
            else:
              raise ValueError
          else:
            net = slim.conv2d(net, growth*2, kernel_size=[3, 3], stride=2, 
              scope='conv1')
          
          net_list = stack_dense_blocks(net,num_classes, blocks, growth, remove_latter_pooling, senet,transition_senet,denseindense, bottleneck, compress,
            stride, rate, drop)

          return net_list,None

def densenet_121(inputs):
  return densenet(inputs, blocks=densenet_utils.networks['densenet_121'], 
    data_name='imagenet')

def densenet_169(inputs):
  return densenet(inputs, blocks=densenet_utils.networks['densenet_169'],
    data_name='imagenet')

def densenet_201(inputs):
  return densenet(inputs, blocks=densenet_utils.networks['densenet_201'],
    data_name='imagenet')

def densenet_265(inputs):
  return densenet(inputs, blocks=densenet_utils.networks['densenet_265'],
    data_name='imagenet')


if __name__ == "__main__":
  x = tf.placeholder(tf.float32, [None, 224, 224, 3])

  net, end_points = densenet_121(x)

  for i in end_points:
    print(end_points[i])
