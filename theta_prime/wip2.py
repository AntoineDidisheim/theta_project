import pandas as pd

import time
from pandarallel import pandarallel
import math
import numpy as np
from parameters import  *
from data import Data
from matplotlib import pyplot as plt
from ml_model import NetworkMean
import os

par = Params()
# Data(par).load_pred_feature(True)

model = NetworkMean(par)

L=os.listdir(model.res_dir)
df = pd.DataFrame()
for l in L:
    print(l)
    df = df.append(pd.read_pickle(model.res_dir+l))


def r2(df_, col='pred'):
    r2_pred = 1 - ((df_['ret1m'] - df_[col]) ** 2).sum() / ((df_['ret1m'] - df_['ret1m'].mean()) ** 2).sum()
    return r2_pred

temp = model.data.load_pred_feature()
temp=temp.dropna()[['permno','date','pred']].rename(columns={'pred':'m_pred'})

df_ = df.dropna().copy().merge(temp)
YEAR=np.sort(df_['date'].dt.year.unique())
R=[]
for y in YEAR:
    print(y)
    ind = df_['date'].dt.year<=y
    r = {'year':y,'NNET':r2(df_.loc[ind,:]),'vilk':r2(df_.loc[ind,:],'vilk'),'m_pred':r2(df_.loc[ind,:],'m_pred')}
    R.append(r)
res = pd.DataFrame(R)
res.index = res['year']
del res['year']
res.plot()
plt.show()

# df_['test'] = df_['pred'].clip(-0.1,0.05)
# df['pred'].max()
# df_['pred'].min()
# df_['vilk']=(1+df_['vilk']*12)**(1/12)-1
r2(df_,'vilk')
r2(df_,'pred')


# df_['year'] = df_['date'].groupby('year')