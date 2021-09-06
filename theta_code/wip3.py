from data import *
from parameters import  *
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os

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
par.data.noise_mw_them
par.update_model_name()

par.data.min_ret = -10000000000000
par.data.max_ret = +10000000000000


a = 'old_res/NEW_them_onlyL10_Lr001Dropout00BS32ActswishOutRange50CVYEAR_BY_YEARLossMAERetRETd2OptCompCrspVOLA_CUBIC/'
b = 'old_res/them_only_with_noise_tsL10_Lr001Dropout00BS32ActswishOutRange50CVYEAR_BY_YEARLossMAERetRETd2OptCompCrspVOLA_CUBIC/'


new = pd.read_pickle(a+'df.p')
old = pd.read_pickle(b+'df.p')
new
(new['pred_norm']-new['ret']).abs().mean()
(old['pred_norm']-old['ret']).abs().mean()
old['pred_norm']

##################
# add other part
##################
id_key = 'permno'
mw = Data(par).marting_wagner_return()
try:
    # if gvkey is main key, add permno to mw
    pr = Data(par).load_all_price()[['permno', 'gvkey']]
    pr['permno'] = pr['permno'].astype(int)
    pr = pr.drop_duplicates()
    mw = mw.merge(pr, how='left')
    id_key = 'gvkey'
except:
    pass
# their MW
them = pd.read_csv('data/MartinWagnerBounds.csv').rename(columns={'id': 'permno'})
them['date'] = pd.to_datetime(them['date'])
them['permno'] = them['permno'].astype(int)
mw = mw.merge(them, how='left')
t = mw[['date', id_key, 'MW', 'mw30']]


df = new.merge(t, how='left')
df['mw30'] = df['mw30'] / 12
# df['mw30'] = df['mw30']*20/30
# their lower bound
them = pd.read_csv(f'{par.data.dir}bench/glb_daily.csv').rename(columns={'id': 'permno'})

them['date'] = pd.to_datetime(them['date'])
them['permno'] = them['permno'].astype(int)
try:
    # again add permno if main is gvkey
    them = them.merge(pr)
except:
    pass

t = them[['date', id_key, 'glb2_D30', 'glb3_D30']]
df = df.merge(t, how='left')
df['glb2_D30'] = df['glb2_D30'] / 12
df['glb3_D30'] = df['glb3_D30'] / 12

# df=df.dropna()


y_bar = df['glb2_D30']
# y_bar = df['mw30']
# y_bar = df['ret'].mean()
df['pred'] = df['pred_norm'].clip(-0.2,0.2)
df['pred_abs'] = df['pred'].abs()
df['max'] = df.groupby('date')['pred_abs'].transform('quantile', 0.99)
df.loc[df['pred_abs'] > df['max'], 'pred'] = np.sign(df.loc[df['pred_abs'] > df['max'], 'pred']) * df.loc[df['pred_abs'] > df['max'], 'max']
ind = (df['ret'] >= -0.5) & (df['ret'] <= 0.5)
df = df.loc[ind, :]

r2_pred = 1 - ((df['ret'] - df['pred']) ** 2).sum() / ((df['ret'] - y_bar) ** 2).sum()
print(r2_pred*100)
