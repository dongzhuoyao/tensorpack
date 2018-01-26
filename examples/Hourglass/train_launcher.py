"""
TRAIN LAUNCHER 

"""

import os
from hourglass_tiny import HourglassModel
from datagen import DataGenerator
from tensorpack.utils import logger
from config import process_config

logger.auto_set_dir()
os.environ['CUDA_VISIBLE_DEVICES'] ='0'


if __name__ == '__main__':
	config = process_config()
	dataset = DataGenerator(config['joint_list'], config['img_directory'], config['training_txt_file'], remove_joints=config['remove_joints'])
	dataset._create_train_table()
	dataset._randomize()
	dataset._create_sets()
	
	model = HourglassModel( training=True,modif=False,dataset=dataset)

	model.generate_model()
	model.training_init(nEpochs=config['nepochs'], epochSize=config['epoch_size'], dataset = None)
	
