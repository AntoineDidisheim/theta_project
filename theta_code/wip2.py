import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from parameters import *
from data import *
from trainer import Trainer
from ml_model import NetworkTheta


##################
# Set parameters
##################
par = Params()
# par.model.layers = [64,32,16]
par.model.layers = [10]
par.model.activation = 'swish'
par.model.batch_size = 32
par.model.learning_rate = 0.01
par.model.E = 10
par.data.val_split = 0.1
par.data.dtype =DataType.OPTION_1
par.model.output_range = 1.2

for d in [DataType.COMP_CRSP_OPTION_1,DataType.OPTION_1,DataType.COMP_CRSP_1]:
    print('#'*100)
    print('Start', d)
    print('#'*100)
    par.data.dtype =d
    res = []
    par.update_model_name()
    par.name
    self = Data(par)
    Data(par).pre_process_all()


