import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from parameters import *
from data import *
from trainer import Trainer
from ml_model import NetworkTheta
import time
import sys

print('#####################################')
print('list', sys.argv)
print('#####################################')

try:
    grid_id = int(sys.argv[1])
except:
    print('Debug mode on local machine')
    grid_id = 0

##################
# Define grid to run
##################


##################
# Set parameters
##################
par = Params()
par.data.dtype = DataType.OPTION_1
par.model.activation = 'swish'
par.model.learning_rate=1e-2
par.model.layers = [10]
par.model.batch_size = 32
par.model.dropout = 0.0
par.model.output_range = 1.2
par.model.E = 5
par.data.val_split = 0.1
res = []

par.update_model_name()
##################
# Create trainer
##################

trainer = Trainer(par)
self = trainer
trainer.create_paper()
# trainer.cv_training()
trainer.create_report_sec()