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
        [10,10]
    ]],
    ['model', 'loss', [
        Loss.MSE, Loss.MAE
    ]],
    ['model', 'batch_size', [
        32
    ]],
    ['model', 'output_range', [
        1,5.0,10.0
    ]]
]

##################
# Set parameters
##################
par = Params()
par.model.tex_name = 'crsp_only_year_year'
par.name_detail = 'Year_year_system_'
par.model.cv = CrossValidation.YEAR_BY_YEAR
par.model.activation = 'swish'
par.model.learning_rate=1e-2
# par.model.batch_size = 32
par.model.E = 5
par.data.val_split = 0.1
par.update_param_grid(gl,grid_id)
par.update_model_name()
par.print_values()
##################
# Create trainer
##################

trainer = Trainer(par)
self = trainer
trainer.create_paper()
trainer.cv_training()
trainer.create_report_sec()