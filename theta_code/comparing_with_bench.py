from data import Data
import pandas as pd
import numpy as np
from statsmodels import api as sm
from matplotlib import pyplot as plt
from parameters import *

# res_dir = 'res/no_cutL10_Lr001Dropout00BS32ActreluOutRange50MinRange-50CVYEAR_BY_YEARLossMSERetRETd3OptCompCrspEXT/'
res_dir = 'res/big_sample_1_L10_Lr001Dropout00BS32ActswishOutRange50MinRange-50CVYEAR_BY_YEARLossMSERetRETd3OptCompCrspEXT/'
par = Params()
par.name_detail = 'big_sample_1_'
par.update_model_name()

df = pd.read_pickle(res_dir + 'df.p')

# for now the return are all normal in the perf report
ret_type_str = 'R'
df['pred'] = df['pred_norm']

# if self.par.data.ret == ReturnType.LOG:
#     ret_type_str = 'log(R)'
# else:
#     ret_type_str  ='R'


df['error_bench'] = (df['ret'] - df['bench']).abs()
df['error_pred'] = (df['ret'] - df['pred']).abs()
df.describe(np.arange(0, 1.05, 0.05)).round(3)

##################
# multiple r2
##################
## add marting wagner
# t=Data(self.par).marting_wagner_return()

'd_3OptCompCrspEXT'

pr = Data(par).load_all_price()[['permno', 'gvkey']]
pr['permno'] = pr['permno'].astype(int)
pr = pr.drop_duplicates()
mw = Data(par).marting_wagner_return()
mw = mw.merge(pr, how='left')
# their MW
them = pd.read_csv('data/MartinWagnerBounds.csv').rename(columns={'id': 'permno'})
them['date'] = pd.to_datetime(them['date'])
them['permno'] = them['permno'].astype(int)
mw = mw.merge(them, how='left')
t = mw[['date', 'gvkey', 'MW', 'mw30']]
df = df.merge(t, how='left')
df['mw30'] = df['mw30'] / 12
df['mw30'] = df['mw30'] * 20 / 30
# their lower bound
them = pd.read_csv(f'{par.data.dir}bench/glb_daily.csv').rename(columns={'id': 'permno'})

them['date'] = pd.to_datetime(them['date'])
them['permno'] = them['permno'].astype(int)
them = them.merge(pr)

t = them[['date', 'gvkey', 'glb2_D30', 'glb3_D30']]
df = df.merge(t, how='left')
df['glb2_D30'] = df['glb2_D30'] / 12
df['glb3_D30'] = df['glb3_D30'] / 12
# df['glb3_D30'] = df['glb3_D30'] * 20 / 30
# df['glb3_D30'] = df['glb3_D30'] * 30 / 20

plt.scatter(df['pred'],df['glb3_D30'],color='k',marker='+')
plt.xlabel('Our estimation')
plt.ylabel('glb3 D30')
plt.grid()
plt.show()



df['month'] = df['date'].dt.year * 100 + df['date'].dt.month
df['one'] = 1.0
df['them'] = df['glb3_D30']
res = []
for mn in np.sort(df['month'].unique()):
    temp = df.loc[df['month']==mn,['them','one','ret','pred']].copy().dropna()

    m_us = sm.OLS(temp['ret'],temp[['pred','one']]).fit()
    m_them = sm.OLS(temp['ret'],temp[['them','one']]).fit()
    #
    # m_us = sm.OLS(temp['ret'],temp[['pred']]).fit()
    # m_them = sm.OLS(temp['ret'],temp[['them']]).fit()

    res.append(
        pd.Series({'beta_us':m_us.params['pred'],'beta_them':m_them.params['them'],
         't_us':m_us.tvalues['pred'],'t_them':m_them.tvalues['them']},name=pd.to_datetime(mn,format='%Y%m'))
    )

res = pd.DataFrame(res)
plt.plot(res.index, res['beta_us'],color='k',label = r'$\beta_{us}$')
plt.plot(res.index, res['beta_them'],color='grey',linestyle='--',label = r'$\beta_{them}$')
plt.grid()
plt.ylabel(r'$\beta$')
plt.legend()
plt.show()

plt.plot(res.index, res['t_us'].abs(),color='k',label = r'$Us$')
plt.plot(res.index, res['t_them'].abs(),color='grey',linestyle='--',label = r'$Them$')
plt.grid()
plt.legend()
plt.ylabel('|T-stats|')
plt.show()

temp = df.loc[:, ['them', 'one', 'ret', 'pred']].copy().dropna()

m_us = sm.OLS(temp['ret'], temp[['pred', 'one']]).fit()
m_them = sm.OLS(temp['ret'], temp[['them', 'one']]).fit()
m_us.summary2()
m_them.summary2()
temp[['ret','them']].corr()
# plt.scatter(df.loc[ind,'glb3_D30'],df.loc[ind,'ret'],color='k',marker='+')
# plt.xlabel('glb3 D30')
# plt.ylabel(r'true return')
# plt.grid()
# plt.show()
#
# plt.scatter(df.loc[ind,'pred'], df.loc[ind,'ret'], color='k', marker='+')
# plt.xlabel('Our estimation')
# plt.ylabel(r'true return')
# plt.grid()
# plt.show()


# df['mw30'] = df['mw30']/12
# df['mw30'] = df['mw30']*20/30


