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
from sklearn.metrics import r2_score

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
par.name_detail = 'rf_fix'
par.model.tex_dir = 'tex/rf_fix'
par.model.cv = CrossValidation.YEAR_BY_YEAR
par.model.activation = 'swish'
par.model.learning_rate = 1e-2
par.model.layers = [10]
par.model.batch_size = 252
par.model.dropout = 0.0


# par.model.layers = [64,32,16]
# par.model.batch_size = 252
# par.model.dropout = 0.1
# par.model.output_range = 1.2
# par.model.out_min=-5.0
# par.model.output_range = 5.0
par.model.out_min=-1.2
par.model.output_range = 3.0
# (self.par.model.output_range-self.par.model.out_min) + self.par.model.out_min
# par.model.out_min=1.0
# par.model.output_range = 2.0
par.model.E = 3
par.data.val_split = 0.1
par.model.loss = Loss.MAE
par.data.opt_smooth = OptSmooth.VOLA_CUBIC
par.data.comp = True
par.data.ret = ReturnType.RET
par.data.min_opt_per_day = 2
par.data.mw =True
par.update_model_name()



par.update_model_name()
par.print_values()


them = pd.read_csv(f'{par.data.dir}bench/glb_daily.csv').rename(columns={'id': 'permno'})
them['date'] = pd.to_datetime(them['date'], format='%Y-%m-%d')
# them['date'].max()
# them[['permno']].astype(int).drop_duplicates().to_csv('data/them_id.txt',index=False,header=False)
#

df = pd.read_csv('data/prc_them.csv')
df.columns = [x.lower() for x in df.columns]

df['S0'] = df['prc']
df = df.rename(columns={'gv_key': 'gvkey', 'cfacpr': 'adj', 'vol': 'total_volume', 'shrout': 'shares_outstanding'})
df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
df = df.sort_values(['permno', 'date']).reset_index(drop=True)
df['S0'] = df['S0'].abs()
df['ret']=pd.to_numeric(df['ret'],errors='coerce')
df['ret_f'] = df.groupby(['permno'])['ret'].shift(-1)


ret_list = ['ret_f']
for T in [19,20,21,22,23]:
    df['S_T'] = df.groupby(['permno'])['S0'].shift(-T)
    df['adj_T'] = df.groupby(['permno'])['adj'].shift(-T)
    df['S_T'] = df['S_T'] * df['adj_T'] / df['adj']
    df[f'r{T}'] = np.exp(np.log(df['S_T'] / df['S0']))-1
    ret_list.append(f'r{T}')


##################
# cum ret method
##################
df['log_ret'] = np.log(df['ret'] + 1)
for h in [19,20,21,22,23]:
    df[f'h{h}'] = df.groupby('permno')['log_ret'].rolling(h).sum().shift(-h).reset_index()['log_ret']
    df[f'h{h}'] = np.exp(df[f'h{h}']) - 1
    ret_list.append(f'h{h}')


f = them.merge(df)
f['pred'] = f['glb2_D30']
# f['ret'] = f['r']
# f['ret'] = np.exp(f['r'])-1

final = []
# ff = f.copy()
for r in ['h20']:
    # f = ff.copy()
    f['ret'] = f[r]
    # print((f['ret']>0.2).mean())
    #
    ind = (f['ret'].abs()<=0.2)
    f=f.loc[ind,:]

    def func(x):
        return pd.qcut(x, 10, labels=False, duplicates='drop').values

    def av_geom(x):
        return (np.prod(1 + x) ** (1 / x.shape[0])) - 1


    # f['port'] = func(f['pred'])
    f['port'] = f.groupby('date')['pred'].transform(func)
    t = f.groupby(['port', 'date'])['ret'].mean().reset_index()
    p = t.groupby('port')['ret'].apply(av_geom)
    # p = t.groupby('port')['ret'].median()
    # p = (1 + p) ** (252/20) - 1
    x= np.log(1+0.097)/np.log(1+p[0])

    p = (1 + p) ** (12) - 1
    print(p)
    p.name = r
    final.append(p)

final=pd.DataFrame(final).T
# final['ret_f']=(final['ret_f']+1)**(252/12) -1
print(final)

final.plot()
plt.show()