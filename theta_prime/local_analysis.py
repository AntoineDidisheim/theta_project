import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from matplotlib import pyplot as plt
from parameters import *
from data import Data
par = Params()
data =Data(par)

def load_final(f):
    d = f'res/{f}/'
    t = pd.DataFrame()
    for l in tqdm(os.listdir(d),f'load {f}'):
        if 'perf_' in l:
            tt=pd.read_pickle(d+l).dropna()
            t = t.append(tt,ignore_index=True)
    return t

mse = 'defaultL64_32_16_Lr0001Dropout00BS512ActreluOutRange05LossMSECssampleFULL'
mae = 'defaultL64_32_16_Lr0001Dropout00BS512ActreluOutRange05LossMAECssampleFULL'

mse = load_final(mse)
mae = load_final(mae)

mse['pred'].quantile(np.arange(0.01,1,0.01)).plot(color='k',label='mse')
mae['pred'].quantile(np.arange(0.01,1,0.01)).plot(color='b',label='mae')
plt.legend()
plt.show()

mse['pred_mae'] = mae['pred']
df = mse.copy()

df['mae']=(df['pred_mae']-df['ret1m']).abs()
df['mse']=(df['pred']-df['ret1m']).abs()

df[['mae','mse']].mean()

df=df.merge(data.load_mw(),how='left')


def r2(df_, col='pred'):
    r2_pred = 1 - ((df_['ret1m'] - df_[col]) ** 2).sum() / ((df_['ret1m'] - 0.0) ** 2).sum()
    return r2_pred
df['mw'] = df['mw'].fillna(0)
df['ret1m'] = df['ret1m'].clip(0.5,-0.5)
r2(df,'vilk')
r2(df,'mw')


them = data.load_vilknoy()

df=df.merge(them,how='left')
r2(df,'vilk')
df
df['glb2_D30'] = ((1+df['glb2_D30'])**(1/12))-1
df['glb3_D30']*=12
df['vilk2'] = np.exp(df['mw'])-1
r2(df,'vilk2')
r2(df,'vilk')


l=-0.2
u = 0.2


1 - ((df['ret1m'].clip(l,u) - df['vilk']) ** 2).sum() / ((df['ret1m'].clip(l,u) - df['mw']) ** 2).sum()




