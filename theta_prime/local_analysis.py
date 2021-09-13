import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from matplotlib import pyplot as plt


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
