"""
TRAIN LAUNCHER 

"""

import configparser,os
from hourglass_tiny import HourglassModel
from datagen import DataGenerator
from tensorpack.utils import logger

logger.auto_set_dir()
os.environ['CUDA_VISIBLE_DEVICES'] ='2'
def process_config(conf_file):
	params = {}
	config = configparser.ConfigParser()
	config.read(conf_file)
	for section in config.sections():
		if section in ['DataSetHG', 'Network', 'Train', 'Validation', 'Saver']:
			for option in config.options(section):
				params[option] = eval(config.get(section, option))
		else:
			logger.warn("invalid section: {}".format(section))
			exit()

	return params


if __name__ == '__main__':
	config = process_config('config.cfg')
	dataset = DataGenerator(config['joint_list'], config['img_directory'], config['training_txt_file'], remove_joints=config['remove_joints'])
	dataset._create_train_table()
	dataset._randomize()
	dataset._create_sets()
	
	model = HourglassModel(nFeat=config['nfeats'], nStack=config['nstacks'], nModules=config['nmodules'], nLow=config['nlow'], outputDim=config['num_joints'], batch_size=config['batch_size'], attention = config['mcam'], training=True, drop_rate= config['dropout_rate'], lear_rate=config['learning_rate'], decay=config['learning_rate_decay'], decay_step=config['decay_step'], dataset=dataset, name=config['name'], logdir_train=config['log_dir_train'], logdir_test=config['log_dir_test'], tiny= config['tiny'], w_loss=config['weighted_loss'], joints= config['joint_list'], modif=False)
	model.generate_model()
	model.training_init(nEpochs=config['nepochs'], epochSize=config['epoch_size'], saveStep=config['saver_step'], dataset = None)
	
