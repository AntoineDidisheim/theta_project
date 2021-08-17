import pandas as pd

import time
from pandarallel import pandarallel
import math
import numpy as np
from parameters import  *
from data import Data

par = Params()
data = Data(par)
mw=data.marting_wagner_return()
gl = pd.read_csv(data.par.data.dir + 'bench/glb_daily.csv')
gl['date'] = pd.to_datetime(gl['date'])
gl = gl[['date', 'id', 'glb3_D30']]
gl.columns = ['date', data.id_col, 'them']

df=data.load_all_price()
print(df.shape)
df=df[['permno','gvkey','date','ret']].drop_duplicates().merge(gl.drop_duplicates())
print(df.shape)
df=df.merge(mw.drop_duplicates(),how='left')
print(df.shape)
df = df.sort_values(['date','permno']).reset_index(drop=True)
del df['gvkey']
df['m_e'] = (df['MW']-df['ret'])**2
df['v_e'] = (df['them'] - df['ret']) ** 2
for err in ['m_e','v_e']:
    TT =[20,60]
    for T in TT:
        df.index = df['date']
        t=df.groupby('permno')[err].rolling(T).agg(['mean','std']).reset_index()
        t.dropna()
        t[err+'_mean']=t.groupby('permno')['mean'].shift(1)
        t[err+'_std']=t.groupby('permno')['std'].shift(1)
        t=t.dropna()
        df = df.reset_index(drop=True)
        df=df.merge(t,how='left')

df.to_pickle('MW_THEM_err_ts.p')

