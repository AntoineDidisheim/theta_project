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
import didipack as didi

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
par.name_detail = 'rf_fix'
par.model.tex_dir = 'tex/rf_fix'
par.model.cv = CrossValidation.EXPANDING
par.model.activation = 'swish'
par.model.learning_rate = 1e-2
par.model.layers = [10]
par.model.batch_size = 252
par.model.dropout = 0.0


# par.model.layers = [64,32,16]
# par.model.batch_size = 252
# par.model.dropout = 0.1
# par.model.output_range = 1.2
# par.model.out_min=-5.0
# par.model.output_range = 5.0
par.model.out_min=-0.0
par.model.output_range = 1.0
# par.model.out_min=-1.2
# par.model.output_range = 3.0

# (self.par.model.output_range-self.par.model.out_min) + self.par.model.out_min
# par.model.out_min=1.0
# par.model.output_range = 2.0
par.model.E = 3
par.data.val_split = 0.1
par.model.loss = Loss.MAE
par.data.opt_smooth = OptSmooth.VOLA_CUBIC
par.data.comp = True
par.data.ret = ReturnType.RET
par.data.min_opt_per_day = 2
par.data.mw =True
par.update_model_name()

par.data.comp = True
par.data.crsp = True


if par.data.comp:
    par.name_detail = 'rf_fix'
    par.model.tex_dir = 'tex/rf_fix'
else:
    par.name_detail = 'rf_fix_opt_only'
    par.model.tex_dir = 'tex/rf_fix_opt_only'

par.update_model_name()
par.print_values()

##################
# Create trainer
##################
# df=Data(par)
# df.load_final()

# try:
#     Data(par).load_final()
# except:
#     Data(par).pre_process_all()
#


trainer = Trainer(par)
self = trainer
# trainer.create_paper()
# trainer.cv_training()
# trainer.create_report_sec()
