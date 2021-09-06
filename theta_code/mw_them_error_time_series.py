import pandas as pd

import time
from pandarallel import pandarallel
import math
import numpy as np
from parameters import  *
from data import Data

par = Params()
par.name_detail = 'surf'
par.model.tex_dir = 'surf'
par.model.cv = CrossValidation.YEAR_BY_YEAR
par.model.activation = 'swish'
par.model.learning_rate = 1e-2
par.model.layers = [10]
par.model.batch_size = 32
par.model.dropout = 0.0
# par.model.output_range = 1.2
par.model.output_range = 5.0
par.model.E = 5
par.data.val_split = 0.1
par.model.loss = Loss.MAE
par.data.opt_smooth = OptSmooth.VOLA_CUBIC
par.data.comp = True
par.data.crsp = True
par.data.ret = ReturnType.RET
par.data.min_opt_per_day = 2
par.data.mw =True
par.update_model_name()


data = Data(par)
# mw=data.marting_wagner_return()
mw = pd.read_csv('data/MartinWagnerBounds.csv').rename(columns={'id': 'permno'})
mw=mw[['date','permno','mw30']]
mw.columns = ['date','permno','MW']
mw['date']=pd.to_datetime(mw['date'])


gl = pd.read_csv(data.par.data.dir + 'bench/glb_daily.csv')
gl['date'] = pd.to_datetime(gl['date'])
gl = gl[['date', 'id', 'glb2_D30']]
gl.columns = ['date', data.id_col, 'them']


df=data.load_all_price(reload=True)


print(df.shape)
df=df[['permno','date','ret']].drop_duplicates().merge(gl.drop_duplicates())
print(df.shape)
df=df.merge(mw.drop_duplicates(),how='left')
print(df.shape)
df = df.sort_values(['date','permno']).reset_index(drop=True)

# del df['gvkey']
# df['MW']/=12
# df['them']/=12

df['m_e'] = (df['MW']-df['ret'])**2
df['v_e'] = (df['them'] - df['ret']) ** 2
df['bench_e'] = (0.0 - df['ret']) ** 2

print(df[['m_e','v_e','bench_e']].mean())
y_bar = 'bench_e'
pred = 'MW'
print(1 - ((df['ret'] - df[pred]) ** 2).sum() / ((df['ret'] - 0.0) ** 2).sum())
T= 20
Q=0.1
err = 'm_e'

df=df[['date','permno','m_e','v_e']]
ind = df[['date','permno']].duplicated(keep=False)
df.loc[ind,:].sort_values(['permno','date'])
df.head()

for err in ['m_e','v_e']:
    TT =[20,60]
    for T in TT:
        print('Before t', df.shape[0]/1e6)
        df.index = df['date']
        t=df.groupby('permno')[err].rolling(T).agg(['mean','std']).reset_index()
        t.dropna()
        t[err+f'_mean_{T}']=t.groupby('permno')['mean'].shift(1)
        t[err+f'_std_{T}']=t.groupby('permno')['std'].shift(1)
        t=t.dropna()
        del t['mean'],t['std']

        df = df.reset_index(drop=True)
        df=df.merge(t,how='left')
        print('After t', df.shape[0]/1e6)

        QQ = [0.1,0.25,0.5,0.75,0.9]
        for Q in QQ:
            df.index = df['date']
            t=df.groupby('permno')[err].rolling(T).quantile(Q).reset_index()
            t.dropna()
            t[err+f'_Q{int(Q*100)}_T{T}']=t.groupby('permno')[err].shift(1)
            t=t.dropna()
            del t[err]
            t = t.drop_duplicates()
            df = df.reset_index(drop=True)


            df=df.merge(t,how='left')
            print(T,Q,df.shape[0]/1e6)
df.to_pickle(par.data.dir+'MW_THEM_err_ts.p')

