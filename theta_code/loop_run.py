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
par.name_detail = 'surf'
par.model.tex_dir = 'surf'
par.model.cv = CrossValidation.YEAR_BY_YEAR
par.model.activation = 'swish'
par.model.learning_rate = 1e-2
par.model.layers = [10]
par.model.batch_size = 32
par.model.dropout = 0.0
# par.model.output_range = 1.2
par.model.output_range = 5.0
par.model.E = 5
par.data.val_split = 0.1
par.model.loss = Loss.MAE
par.data.opt_smooth = OptSmooth.VOLA_CUBIC
par.data.comp = True
par.data.crsp = True
par.data.ret = ReturnType.RET
par.data.min_opt_per_day = 2
par.data.mw =True

par.update_model_name()

par.data.min_ret = -10000000000000
par.data.max_ret = +10000000000000

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
var_subset = [
    ['MW', 'them', 'theta_v'],
    ['MW', 'them', 'theta_v'],
    ['MW', 'them', 'theta_v'],
    ['MW', 'them', 'theta_v']
    # ['iv80', 'iv90', 'iv100', 'iv110', 'iv120'],
    # ['iv80', 'iv90', 'iv100', 'iv110', 'iv120']
]

name_list = [
    # 'them_only_with_noise_ts'
    'NO_DIV_noise_them_',
    'NEW_them_only_with_noise_ts',
    # 'NEW_them_only_with_noise_0_1_',
    'NEW_them_only_with_noise_OLD_'
]

tf.random.set_seed(1234)
np.random.seed(1234)

for i in range(len(name_list)):
    par.name_detail = name_list[i]
    par.model.tex_dir = f'tex/{name_list[i]}'
    par.data.var_subset  = var_subset[i]

    # if i == 1:
    #     par.model.layers = [64, 32, 16]
    # else:
    #     par.model.layers = [10]

    par.update_model_name()
    par.print_values()

    print('='*500)
    print('Starts,', name_list[i])
    print('='*500)

    trainer = Trainer(par)
    self = trainer
    trainer.create_paper()
    # trainer.cv_training()
    trainer.create_report_sec()
