import os
import sys
sys.setrecursionlimit(10000)
import logging
import getpass
username = getpass.getuser()
import numpy as np

#from MPIAllJoints import *

class Config:
    def ensure_dir(path):
        """create directories if *path* does not exist"""
        if not os.path.isdir(path):
            os.makedirs(path)

    program_name = 'skeleton'
    cur_dir = os.path.dirname(__file__)
    abs_dir = cur_dir
    this_dir = cur_dir.split('/')[-1]
    root_dir = os.path.join(cur_dir, '..', '..', '..')

    # The log
    log_folder = os.path.join(root_dir, 'logs', username+'.'+this_dir)
    log_dir_link = log_folder
    ensure_dir(log_dir_link)
    if os.path.exists('log'):
        os.remove('log')
    os.system('ln -s {} log'.format(log_dir_link))
    link_log_file = os.path.join(log_dir_link, 'exp.log')

    skeleton_root_dir = os.path.join(cur_dir, '..', '..', '..')
    img_path = os.path.join(skeleton_root_dir, 'data', 'MPIHP', 'images')

    sys.path.insert(0, root_dir)
    #from _init_ import *
    sys.path.insert(0, os.path.join(skeleton_root_dir, 'data'))

###################################### project info #################################################
    proj_name = this_dir
    output_dir = os.path.join(log_folder, 'snapshot')
    ensure_dir(output_dir)


    """ Tensorboard Settings. Generally, you don't have to change these settings"""
###############################################################################################################################################
    # control Tensorboard starting. Normally, you can set monitor_enable = True to start Tensorboard
    tensorboard_enable=True
    tensorboard_logger_log_dir = os.path.abspath(log_dir_link + '/tb_log_dir/')
    link_tensorboard_logger_log_dir = os.path.abspath(log_dir_link + '/tb_log_dir')
    tensorboard_port = 8080

    # NOTE: If you open the following controls, the training speed will be hury baddly !!!!!!
    watch_params = False
    watch_grads  = False
    watch_interval = 200
###############################################################################################################################################


    exp_name = username + '.' + proj_name

###################################### data preparation ##############################################
    nr_skeleton=16

    imgExtXBorder = 0.10
    imgExtYBorder = 0.15

    pixel_means = np.array([[[102.9801, 115.9465, 122.7717]]]) # BGR
    pixel_norm = True
    data_shape = (256, 256) #height, width
    output_shape = (64, 64) #height, width

    dpflow_name = username + '.' + proj_name
    dpflow_enable = True
    dpflow_buffer= 10
    dpflow_train_thread = 10

###################################### train setup params ############################################
    nr_epochs = 1000
    epoch_size = 30000 # include flip * 2, aug * 4, batch * 16

    lr = 0.001
    optimizer = 'adam'
    lr_accum_step = 1
    batch_size = 16 # batch size in per gpu i.e. the first dim of inputs
    lr_dec_epoch = 30
    lr_gamma = 0.5

    weight_decay = 1e-4

###################################### train model ###################################################
    bn_training = True
    ResNet = False
    brain_model = None
    init_model = '/unsullied/sharefs/yugang/ceph-dataset/model_zoos/brain_model_zoo/zxy/res50_hyper/bottleneck50_hypernet.brainmodel'

    my_batch_normalization = False

###################################### extras #######################################################
# data aug
    data_aug = True
    nr_aug_copies = 1+1+2 # origin, flip, rotation

# related to Megbrain
    enforce_var_shape = True
    def __str__(self):
        atts = dir(self)
        ret = ''
        for a in atts:
            if '__' not in a:
                ret += a + '\t: ' + str(getattr(self, a)) + '\n'
        return ret
config = Config()
