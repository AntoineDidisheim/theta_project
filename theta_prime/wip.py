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
from statsmodels import api as sm
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
import matplotlib
import pylab


if 'nv-' in socket.gethostname():
    matplotlib.use('Agg')

params = {'axes.labelsize': 14,
          'axes.labelweight': 'normal',
          'xtick.labelsize': 12,
          'ytick.labelsize': 12,
          'axes.titlesize': 12}
pylab.rcParams.update(params)

par = Params()
par.name_detail='PostVac'
par.data.cs_sample = CSSAMPLE.FULL
# par.model.layers = [100,100,100]
par.model.layers = [32,16,8]
par.model.dropout = 0.0
par.model.batch_size = 256
par.model.output_range = 0.5
par.model.learning_rate = 0.001
par.model.loss = Loss.MSE
par.model.E = 10
par.data.H = 20

name = 'PostVacL100_100_100_Lr0001Dropout02BS512ActreluOutRange05pLossMSECssampleFULL'
par.name =  name


data = Data(par)
data.load_internally()


def tr_func(x):
    t = ''
    if x == 'mean_pred':
        t = 'mean'
    elif x == 'median_pred':
        t = 'median'
    else:
        nb_days = x.split('_')[-1]
        predictor = x.split('err_')[1].split('_')[0]
        agg_id = x.split('err_' + predictor + '_')[-1].split('_')[0]
        if 'mean' in agg_id:
            agg = 'average absolute error'
        if 'std' in agg_id:
            agg = 'variance absolute error'
        if 'Quantile' in agg_id:
            if '0.75' in agg_id:
                agg = 'upper quartile absolute error'
            if '0.25' in agg_id:
                agg = 'lower quartile absolute error'
        t = f'{predictor} predictor | {nb_days} days {agg}'
    return t

m_col=['mean_pred'] + [x for x in data.x_df.columns if 'err_mean' in x]
corr = data.x_df[m_col].corr()
corr.index = [tr_func(x) for x in corr.index]
corr.columns = [tr_func(x) for x in corr.columns]

temp = data.x_df[['mean_pred']]
temp['constant'] = 1.0
temp['y'] = data.label_df['ret1m']
temp = temp.dropna()
m=sm.OLS(temp['y'],temp[['constant','mean_pred']]).fit()
m.summary2()
