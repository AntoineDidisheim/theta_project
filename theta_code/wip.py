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
##################
# Create trainer
##################
# df=Data(par)
# df.load_final()

# try:
#     Data(par).load_final()
# except:
#     Data(par).pre_process_all()
#


trainer = Trainer(par)

self = trainer

##################
# inside trainer
##################

df = pd.read_pickle(self.res_dir + 'df.p')






# for now the return are all normal in the perf report
ret_type_str = 'R'
df['pred'] = df['pred_norm'].clip(-0.2,0.2)

df['bench'] = df['bench'].clip(-0.2,0.2)




# if self.par.data.ret == ReturnType.LOG:
#     ret_type_str = 'log(R)'
# else:
#     ret_type_str  ='R'


df['error_bench'] = (df['ret'] - df['bench']).abs()
df['error_pred'] = (df['ret'] - df['pred']).abs()
df.describe(np.arange(0,1.05,0.05)).round(3)





##################
# multiple r2
##################
## add marting wagner
# t=Data(self.par).marting_wagner_return()
id_key = 'permno'
mw = Data(self.par).marting_wagner_return()
try:
    # if gvkey is main key, add permno to mw
    pr = Data(self.par).load_all_price()[['permno', 'gvkey']]
    pr['permno'] = pr['permno'].astype(int)
    pr = pr.drop_duplicates()
    mw = mw.merge(pr, how='left')
    id_key='gvkey'
except:
    pass
# their MW
them = pd.read_csv('data/MartinWagnerBounds.csv').rename(columns={'id': 'permno'})
them['date'] = pd.to_datetime(them['date'])
them['permno'] = them['permno'].astype(int)
mw = mw.merge(them, how='left')
t=mw[['date',id_key,'MW','mw30']]
df = df.merge(t, how='left')
df['mw30'] = df['mw30']/12
df['mw30'] = df['mw30']*20/30
# their lower bound
them = pd.read_csv(f'{self.par.data.dir}bench/glb_daily.csv').rename(columns={'id': 'permno'})

them['date'] = pd.to_datetime(them['date'])
them['permno'] = them['permno'].astype(int)
try:
    # again add permno if main is gvkey
    them = them.merge(pr)
except:
    pass

t=them[['date',id_key,'glb2_D30','glb3_D30']]
df = df.merge(t, how='left')
df['glb2_D30'] = df['glb2_D30'] / 12
df['glb3_D30'] = df['glb3_D30'] / 12
# df['glb3_D30'] = df['glb3_D30'] *20/30

# plt.scatter(df['pred'],df['glb3_D30'],color='k',marker='+')
# plt.xlabel('Our estimation')
# plt.ylabel('glb3 D30')
# plt.grid()
# plt.show()
#
# plt.scatter(df['bench'],df['glb3_D30'],color='k',marker='+')
# plt.xlabel(r'$\theta = 1.0$')
# plt.ylabel('glb3 D30')
# plt.grid()
# plt.show()


df['month'] = df['date'].dt.year*100+df['date'].dt.month



try:
    t=Data(self.par).historical_theta()
    df = df.merge(t[['date',id_key,'pred']].rename(columns={'pred':'hist_theta'}), how='left')
    t = Data(self.par).load_all_price(False)
    t['year'] = t['date'].dt.year
    df['year'] =  df['date'].dt.year
    t=t.groupby('year')['ret'].mean().reset_index()
    t[r'$\bar{MKT}_{t-1}$']=t['ret'].shift()
    t = t.rename(columns={'ret':r'$\bar{MKT}_{t}$'})
    overall_average =  Data(self.par).load_all_price()['ret'].mean()
    df=df.merge(t)
except:
    df[r'$\bar{MKT}_{t-1}$'] = np.nan
    df[r'hist_theta'] = np.nan
    overall_average = Data(self.par).load_all_price()['ret'].mean()

try:
    t = Data(self.par)
    t.load_final()
    t.label_df[r'$\beta_{i,t} \bar{MKT}_{t}$'] = (t.p_df['beta_monthly']*t.p_df['mkt-rf'])/100
    df=df.merge(t.label_df)
except:
    df[r'$\beta_{i,t} \bar{MKT}_{t}$'] = np.nan

def r2(df_,y_bar, name='NNET'):
    try:
        if np.sum(pd.isna(y_bar))>0:
            df_ = df_.loc[~pd.isna(y_bar),:]

        r2_pred = 1 - ((df_['ret'] - df_['pred']) ** 2).sum() / ((df_['ret'] - y_bar) ** 2).sum()
        r = (pd.Series({name: r2_pred})*100).round(2)
    except:
        r = np.nan

    return r

if 'hist_theta' not in df.columns:
    df['hist_theta']=np.nan
if '$\\bar{MKT}_{t}$' not in df.columns:
    df['$\\bar{MKT}_{t}$']=np.nan
df.describe(np.arange(0,1.05,0.05))
ind = (df['ret']>=-0.5) & (df['ret']<=0.5)
df = df.loc[ind,:]
def get_all_r(df):
    r=[
        r2(df,(1.06)**(1/12)-1,r'6\% premium'),
        r2(df,df['MW'],'Martin Wagner'),
        r2(df,df['mw30'],'Martin Wagner downloaded'),
        r2(df,0.0, r'$R=0.0$'),
        r2(df, df['hist_theta'],r'historical $\theta$'),
        r2(df, df['bench'],r'$\theta=1.0$'),
        r2(df, df[r'$\bar{MKT}_{t}$'],r'$\bar{MKT}_{t}$'),
        r2(df, df[r'$\bar{MKT}_{t-1}$'],r'$\bar{MKT}_{t-1}$'),
        r2(df, overall_average,r'$\bar{MKT}$'),
        r2(df, df[r'$\beta_{i,t} \bar{MKT}_{t}$'],r'$\beta_{i,t} \bar{MKT}_{t}$'),
        r2(df, df[r'glb2_D30'],r'Vilkny glb2 D30'),
        r2(df, df[r'glb3_D30'],r'Vilkny glb3 D30')
    ]
    return pd.concat(r).sort_values()

df['year'] = df['date'].dt.year

t=df.groupby('year').apply(lambda x: get_all_r(x)).reset_index()
t.columns = ['Year', 'Type', r'$R^2$']
t = t.pivot(columns='Year', index='Type')
t['All'] = get_all_r(df)
t = t.sort_values('All')
tt = df.groupby('year')['date'].count()
tt['All'] = df.shape[0]
t = t.T
t['nb. obs'] = tt.values
t = t.T
print(t)



plt.scatter(df['glb2_D30'], df['pred'],color='k',marker='+')
plt.xlabel('Vilknoy')
plt.ylabel('NNET pred')
plt.show()


plt.scatter(df['theta'], df['pred'],color='k',marker='+')
plt.xlabel('theta')
plt.ylabel('NNET pred')
plt.show()

df[['glb2_D30','pred']].quantile(np.arange(0.001,1,0.001)).plot()
plt.xlabel('Percentile')
plt.ylabel('Return')
plt.show()

df['year'] = df['date'].dt.year
temp =df.loc[df['year']==2018,:]
temp[['glb2_D30','pred']].quantile(np.arange(0.001,1,0.001)).plot()
plt.xlabel('Percentile')
plt.ylabel('Return')
plt.show()

df.groupby('year')['theta'].mean()


df = df.sort_values(['permno','date']).reset_index(drop=True)
df['theta_l'] = df.groupby('permno')['theta'].shift(-1)
df['one'] = 1.0
temp = df[['date','permno','one','theta','theta_l']].dropna()
temp['theta_m']=temp.groupby('permno')['theta'].transform('mean')
temp['year'] = temp['date'].dt.year
temp['month'] = temp['date'].dt.year*100 + temp['date'].dt.month
temp['theta_y']=temp.groupby('year')['theta'].transform('mean')
temp['theta_ym']=temp.groupby(['year','permno'])['theta'].transform('mean')
temp['theta_mm']=temp.groupby(['month','permno'])['theta'].transform('mean')

r2_score(temp['theta'],temp['theta_m'])
r2_score(temp['theta'],temp['theta_y'])
r2_score(temp['theta'],temp['theta_ym'])
r2_score(temp['theta'],temp['theta_mm'])
r2_score(temp['theta'],temp['theta_l'])

temp['theta_ym'].hist(bins=100)
plt.show()


sm.OLS(temp['theta'],temp[['theta_l','one']]).fit().summary2()


