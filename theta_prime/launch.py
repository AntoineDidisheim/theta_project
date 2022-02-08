import numpy as np
import pandas as pd
from parameters import *
from trainer import Trainer
from data import  Data
import socket
import pandas as pd
import time
from pandarallel import pandarallel
import math
import numpy as np
from parameters import  *
from data import Data
from matplotlib import pyplot as plt
from ml_model import NetworkMean
import didipack as didi
import os
from tqdm import tqdm
import seaborn as sns
from scipy.stats import pearsonr
import shutil

if 'nv-' in socket.gethostname():
    import matplotlib
    matplotlib.use('Agg')

par = Params()
# par.name_detail='WithRET_'
par.name_detail='Simple3_'
# par.name_detail='Subset_'
par.data.cs_sample = CSSAMPLE.FULL
par.model.layers = [64,32,16]
par.model.dropout = 0.2
par.model.output_range=0.2
par.model.learning_rate = 0.001
par.model.loss = Loss.MSE
par.data.H = 20
par.model.batch_normalization = False
par.model.regulator = False
par.data.inter_quartile_version=True

# C=['mean_pred', 'err_mean_mean_20', 'err_mean_std_20', 'err_mean_Quantile0.25_20', 'err_mean_Quantile0.75_20', 'err_mean_mean_180', 'err_mean_std_180', 'err_mean_Quantile0.25_180', 'err_mean_Quantile0.75_180', 'err_mean_mean_252', 'err_mean_std_252', 'err_mean_Quantile0.25_252', 'err_mean_Quantile0.75_252', 'err_true_ret_mean_20', 'err_true_ret_std_20', 'err_true_ret_Quantile0.25_20', 'err_true_ret_Quantile0.75_20', 'err_true_ret_mean_180', 'err_true_ret_std_180', 'err_true_ret_Quantile0.25_180', 'err_true_ret_Quantile0.75_180', 'err_true_ret_mean_252', 'err_true_ret_std_252', 'err_true_ret_Quantile0.25_252', 'err_true_ret_Quantile0.75_252']
# C=['mean_pred', 'err_mean_mean_20', 'err_mean_std_20', 'err_mean_Quantile0.25_20', 'err_mean_Quantile0.75_20', 'err_mean_mean_180', 'err_mean_std_180', 'err_mean_Quantile0.25_180', 'err_mean_Quantile0.75_180', 'err_mean_mean_252', 'err_mean_std_252', 'err_mean_Quantile0.25_252', 'err_mean_Quantile0.75_252', 'err_true_ret_mean_20', 'err_true_ret_std_20', 'err_true_ret_Quantile0.25_20', 'err_true_ret_Quantile0.75_20', 'err_true_ret_mean_180', 'err_true_ret_std_180', 'err_true_ret_Quantile0.25_180', 'err_true_ret_Quantile0.75_180', 'err_true_ret_mean_252', 'err_true_ret_std_252', 'err_true_ret_Quantile0.25_252', 'err_true_ret_Quantile0.75_252']
# for c in C:
#     if 'std' in c:
#         C.remove(c)
#
# par.data.var_subset = C
# par.data.var_subset = ['err_true_ret_median_20', 'err_true_ret_Quantile0.25_20',
#                        'err_true_ret_Quantile0.75_20', 'err_true_ret_median_180',
#                        'err_true_ret_Quantile0.25_180', 'err_true_ret_Quantile0.75_180',
#                        'err_true_ret_median_252', 'err_true_ret_Quantile0.25_252', 'err_true_ret_Quantile0.75_252']
#
# t = [x.replace('err_true_ret','err_median') for x in par.data.var_subset]
# par.data.var_subset = par.data.var_subset + t

par.update_model_name()



data = Data(par)

trainer = Trainer(par)
self = trainer
trainer.launch_training_expanding_window()
trainer.create_paper()



# ['mean_pred', 'err_mean_mean_20', 'err_mean_std_20', 'err_mean_Quantile0.25_20', 'err_mean_Quantile0.75_20', 'err_mean_mean_180', 'err_mean_std_180', 'err_mean_Quantile0.25_180', 'err_mean_Quantile0.75_180', 'err_mean_mean_252', 'err_mean_std_252', 'err_mean_Quantile0.25_252', 'err_mean_Quantile0.75_252', 'err_true_ret_mean_20', 'err_true_ret_std_20', 'err_true_ret_Quantile0.25_20', 'err_true_ret_Quantile0.75_20', 'err_true_ret_mean_180', 'err_true_ret_std_180', 'err_true_ret_Quantile0.25_180', 'err_true_ret_Quantile0.75_180', 'err_true_ret_mean_252', 'err_true_ret_std_252', 'err_true_ret_Quantile0.25_252', 'err_true_ret_Quantile0.75_252']