import pandas as pd
import time
from pandarallel import pandarallel
import math
import numpy as np
from parameters import *
from data import Data
from matplotlib import pyplot as plt
from ml_model import NetworkMean
par = Params()
par.data.H = 120
data=Data(par)

reload = False
df=data.load_vilknoy().merge(data.load_mw())
df=df.merge(data.load_all_price())

# df =data.load_all_price()
print(df['date'].min())

def r2(df_, col='vilk'):
    temp = df_[['ret1m',col,'mw']].copy().dropna()

    r2_pred = 1 - ((temp['ret1m'] - temp[col]) ** 2).sum() / ((temp['ret1m'] - temp['mw']) ** 2).sum()
    return r2_pred*100

ind = df['date'].dt.year>=2000
r2(df.loc[ind,:])

df['year'] = df['date'].dt.year


print(df.groupby('year').apply(r2))
print(r2(df.loc[:,:]))

df[['ret1m','mw','vilk']]

