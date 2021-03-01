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

par.name_detail = 'no_strike_arbitrage_AND_low_spread_'
# par.data.dtype = DataType.OPTION_1
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
data = Data(par)
opt=data.load_opt()

##################
# add cleaning criteria
##################

# seems to do shit
opt['sr'] = opt['strike']/opt['S0']
ind_1 = opt.groupby(['gvkey','date'])['sr'].transform('min') > (1-0.2*np.sqrt(1/12))
ind_2 = opt.groupby(['gvkey','date'])['sr'].transform('max') < (1+0.2*np.sqrt(1/12))
opt['t'] = ind_1 | ind_2
opt['t'].mean()

# # max distance
# opt = opt.sort_values(['gvkey','date','strike']).reset_index(drop=True)
# opt['t']=opt.groupby(['gvkey','date'])['opt_price'].diff().abs()
# opt['t']=opt.groupby(['gvkey','date'])['t'].transform('max')
# ind=(opt['t']>opt['S0']*0.2) & (opt['t']>10)
# opt['t'] = ind

df = pd.read_pickle(trainer.res_dir+'df.p')

df['error_bench'] = (df['ret'] - df['bench']).abs()
df['error_pred'] = (df['ret'] - df['pred']).abs()
df['d_error'] = (df['error_bench']-df['error_pred'])
df['d_error_abs'] = (df['error_bench']-df['error_pred']).abs()




plt.figure(figsize=(6.4*2,4.8*5))
df.sort_values('d_error').tail(10)
for i in range(10):
    id = df.sort_values('d_error').iloc[-i,:]
    # id = df.sort_values('d_error_abs').iloc[i,:]
    print('#'*50)
    print(i)
    print('#'*50)

    ind=(opt['gvkey'] == id['gvkey']) & (opt['date']==id['date'])
    t=opt.loc[ind,:].sort_values('strike')
    print(t[['S0','cp','strike','opt_price']])

    plt.subplot(5,2,i+1)
    plt.plot(t['strike'],t['opt_price'])
    plt.vlines(t['S0'].iloc[0],t['opt_price'].min(),t['opt_price'].max())
    plt.xlabel(str(i)+ str(t['t'].iloc[0]))



plt.show()


    ##################
    # develop the additional cleaning level
    ##################

