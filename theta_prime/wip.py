import pandas as pd

import time
from pandarallel import pandarallel
import math
import numpy as np
from parameters import  *
from data import Data
from matplotlib import pyplot as plt
from ml_model import NetworkMean
par = Params()
data=Data(par)

df=data.load_vilknoy().merge(data.load_mw())
df=df.merge(data.load_all_price())



def r2(df_, col='vilk'):
    r2_pred = 1 - ((df_['ret1m'] - df_[col]) ** 2).sum() / ((df_['ret1m'] - df_['mw']) ** 2).sum()
    return r2_pred*100

# df['mw'] = df['mw'].fillna(0)
ind = df['date'].dt.year>=2000
r2(df.loc[ind,:])
r2(df.loc[:,:])



