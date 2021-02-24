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

gl = [
    ['model', 'layers', [
        [10],
        [64,32,16]
    ]],
    # ['model', 'opti', [
    #     Optimizer.ADAM
    # ]],
    ['model', 'batch_size', [
        16,
        256
    ]],
    ['model', 'learning_rate', [
        1e-3,
        0.5e-4
    ]],
    ['data', 'dtype', [
        DataType.COMP_CRSP_OPTION_1,
        DataType.CRSP_OPTION_1,
        DataType.OPTION_1
    ]]
]

##################
# Set parameters
##################
par = Params()
par.model.activation = 'swish'
par.model.batch_size = 32
par.model.E = 10
par.data.val_split = 0.1
res = []
par.update_param_grid(gl, grid_id)
par.update_model_name()
##################
# Create trainer
##################

trainer = Trainer(par)
self = trainer
trainer.create_paper()
trainer.cv_training()
trainer.create_report_sec()