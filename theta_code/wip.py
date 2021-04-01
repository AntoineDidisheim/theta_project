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


##################

# Set parameters
##################
par = Params()
par.name_detail = 'beta05_'
par.model.tex_dir = 'beta05_'
par.model.cv = CrossValidation.YEAR_BY_YEAR
par.model.activation = 'swish'
par.model.learning_rate = 1e-2
par.model.layers = [10]
par.model.batch_size = 32
# par.model.layers = [64,32,16]
# par.model.batch_size = 256

par.model.dropout = 0.0
# par.model.dropout = 0.4
# par.model.output_range = 1.2
par.model.output_range = 5.0
par.model.out_min = -5.0
par.model.E = 5
par.data.val_split = 0.1
par.model.loss = Loss.MSE
par.data.opt_smooth = OptSmooth.EXT
par.data.min_opt_per_day = 10
par.data.comp = True
par.data.ret = ReturnType.LOG


par.update_model_name()
par.print_values()

df = Data(par).load_opt()
rf=Data(par).load_rf()
rf.columns=['date','rf']
df=df.merge(rf)
##################
# main
##################

def BlackScholes_price(S, r, sigma, K):
    dt = 28 / 365
    Phi = stats.norm(loc=0, scale=1).cdf

    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * dt) / (sigma * np.sqrt(dt))
    d2 = d1 - sigma * np.sqrt(dt)

    pr = S * Phi(d1) - K * np.exp(-r * dt) * Phi(d2)
    pr_put = K * np.exp(-r * dt) * Phi(-d2) - S * Phi(-d1)
    pr[S > K] = pr_put[S > K]
    return pr
df.head()
df
p=BlackScholes_price(df['S'],df['rf'],df['impl_volatility'],df['strike'])
df['pr'] =p
df['true_p'] = (df['o_ask']+df['o_bid'])/2

df['err']=(df['pr']-df['true_p']).abs()/df['true_p']
df=df.loc[df['delta'].abs()<=0.5,:]

df['err'].describe(np.arange(0,1.05,0.05)).round(2)