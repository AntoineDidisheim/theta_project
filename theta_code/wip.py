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
par.data.opt_smooth = OptSmooth.INT
par.data.min_opt_per_day = 15
par.data.comp = True
par.data.ret = ReturnType.LOG


par.update_model_name()
par.print_values()


data = Data(par)

df = data.historical_theta()

df=df.loc[~df['na'],:]


df['theta'].hist(bins=100)
plt.show()
df.dtypes
df['theta'] = pd.to_numeric(df['theta'])
df.groupby('date')['theta'].mean().rolling(4).mean().plot()
plt.show()