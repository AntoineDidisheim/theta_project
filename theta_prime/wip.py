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


shap_col=['permno', 'date', 'ticker', 'ret1m', 'pred', 'mean_pred', 'err_mean_mean_20', 'err_mean_std_20', 'err_mean_Quantile0.25_20', 'err_mean_Quantile0.75_20', 'err_mean_mean_180', 'err_mean_std_180', 'err_mean_Quantile0.25_180', 'err_mean_Quantile0.75_180', 'err_mean_mean_252', 'err_mean_std_252', 'err_mean_Quantile0.25_252', 'err_mean_Quantile0.75_252', 'median_pred', 'err_median_mean_20', 'err_median_std_20', 'err_median_Quantile0.25_20', 'err_median_Quantile0.75_20', 'err_median_mean_180', 'err_median_std_180', 'err_median_Quantile0.25_180', 'err_median_Quantile0.75_180', 'err_median_mean_252', 'err_median_std_252', 'err_median_Quantile0.25_252', 'err_median_Quantile0.75_252', 'err_true_ret_mean_20', 'err_true_ret_std_20', 'err_true_ret_Quantile0.25_20', 'err_true_ret_Quantile0.75_20', 'err_true_ret_mean_180', 'err_true_ret_std_180', 'err_true_ret_Quantile0.25_180', 'err_true_ret_Quantile0.75_180', 'err_true_ret_mean_252', 'err_true_ret_std_252', 'err_true_ret_Quantile0.25_252',
       'err_true_ret_Quantile0.75_252']
x = 'err_true_ret_Quantile0.75_252'

def tr_func(x):
    t = ''
    if x == 'mean_pred':
        t = 'mean'
    elif x == 'median_pred':
        t = 'median'
    else:
        nb_days = x.split('_')[-1]
        predictor = x.split('err_')[1].split('_')[0]
        if predictor == 'true':
            predictor = 'return'
        agg_id = x.split('err_' + predictor + '_')[-1].split('_')[0]
        if agg_id == 'err':
            agg_id=x.split('err_true_ret_')[-1].split('_')[0]
        if 'mean' in agg_id:
            agg = 'average absolute error'
        elif 'std' in agg_id:
            agg = 'variance absolute error'
        else:
            agg = 'return'
        if 'Quantile' in agg_id:
            if '0.75' in agg_id:
                agg = 'upper quartile absolute error'
            if '0.25' in agg_id:
                agg = 'lower quartile absolute error'
        t = f'{predictor} predictor | {nb_days} days {agg}'
    return t
tr = []
print(shap_col)
for x in shap_col:
    if x in shap_col[5:]:
        tr.append(tr_func(x))
    else:
        tr.append(x)

len(tr)
len(np.unique(tr))