from dataset import DataIter
from config import config
import sys
sys.path.insert(0, '../../../data/MPIHP/')
from MPIAllJoints import MPIJoints

d = MPIJoints()
train_data, validation_data = d.load_data()
di = DataIter("train", train_data, config.data_shape)
while True:
    minibatch = next(di)
    print("ok")