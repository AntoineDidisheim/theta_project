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
par.name_detail = 'back'
par.model.tex_dir = 'back'
par.model.cv = CrossValidation.YEAR_BY_YEAR
par.model.activation = 'swish'
par.model.learning_rate = 1e-2
par.model.layers = [10]
par.model.batch_size = 32
# par.model.layers = [64,32,16]
# par.model.batch_size = 256

par.model.dropout = 0.0
# par.model.dropout = 0.4
par.model.output_range = 1.2
# par.model.output_range = 5.0
par.model.E = 5
par.data.val_split = 0.1
par.model.loss = Loss.MSE
par.data.opt_smooth = OptSmooth.EXT
par.data.min_opt_per_day = 10
par.data.comp = True
par.data.ret = ReturnType.LOG


par.update_model_name()
par.print_values()


data = Data(par)

df = data.load_opt()


def BlackScholes_price(S, r, sigma, K):
    dt = 28 / 365
    Phi = stats.norm(loc=0, scale=1).cdf

    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * dt) / (sigma * np.sqrt(dt))
    d2 = d1 - sigma * np.sqrt(dt)

    pr = S * Phi(d1) - K * np.exp(-r * dt) * Phi(d2)
    pr_put = K * np.exp(-r * dt) * Phi(-d2) - S * Phi(-d1)
    pr[S > K] = pr_put[S > K]
    return pr


df  =df[df['delta'].abs()<=0.5]
df  =df.reset_index(drop=True)
S = df['S']
r = 0.01
IV = df['impl_volatility']
K = df['strike']
PRICE = BlackScholes_price(S, r, IV, K)
df['pred_pr'] =PRICE
t=(df['o_ask']+df['o_bid'])/2
df['mid_p'] = t
tt=((t-PRICE).abs()/t)
tt.describe(np.arange(0,1.05,0.05)).round(2)
tt[df['delta'].abs()<=0.5].describe(np.arange(0,1.05,0.05)).round(2)
temp=df.loc[tt>0.2,:]

df = df[(df['delta'].abs()<=0.5)]

df['impl_volatility'].max()
(df.groupby(['gvkey','date'])['strike'].nunique()<10).mean()
df.groupby(['gvkey','date'])['strike'].nunique().describe(np.arange(0,1.05,0.05)).round(2)


# tt=tt[df['delta'].abs()<=1].describe(np.arange(0,1.05,0.05)).round(2)

# temp=df[df['delta'].abs()<=1]
# temp.loc[t == tt.max(),:]
