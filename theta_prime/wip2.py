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
import didipack as didi


par = Params()
fig_save_dir = 'res/fig/'
os.makedirs(fig_save_dir,exist_ok=True)
# Data(par).load_pred_feature(True)

model = NetworkMean(par)

L=os.listdir(model.res_dir)
df = pd.DataFrame()
for l in L:
    print(l)
    df = df.append(pd.read_pickle(model.res_dir+l))


def r2(df_, col='pred'):
    r2_pred = 1 - ((df_['ret1m'] - df_[col]) ** 2).sum() / ((df_['ret1m'] - 0.0) ** 2).sum()
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

##################
# cummulative r2
##################
d = pd.to_datetime(res.index,format='%Y')
for i, c in enumerate(['NNET','vilk','m_pred']):
    plt.plot(d,res[c],color=didi.DidiPlot.COLOR[i],linestyle=didi.DidiPlot.LINE_STYLE[i],label=c)
plt.grid()
plt.xlabel('Year')
plt.ylabel(r'Cummulative $R^2$')
plt.legend()
plt.tight_layout()
plt.savefig(fig_save_dir+'cummulative_r2.png')
plt.show()

##################
# year per year
##################
df_['year'] = df_['date'].dt.year
for i, c in enumerate(['pred','vilk']):
    t=df_.groupby('year').apply(lambda x: r2(x,col=c))
    plt.plot(d,t,color=didi.DidiPlot.COLOR[i],linestyle=didi.DidiPlot.LINE_STYLE[i],label=c if c != 'pred' else 'NNET')
plt.grid()
plt.xlabel('Year')
plt.ylabel(r'Year per Year $R^2$')
plt.legend()
plt.tight_layout()
plt.savefig(fig_save_dir+'year_r2.png')
plt.show()


##################
# stock by stock
##################
S =[]
df_['year'] = df_['date'].dt.year
for i, c in enumerate(['pred','vilk']):
    t=df_.groupby('permno').apply(lambda x: r2(x,col=c))
    plt.hist(t,color=didi.DidiPlot.COLOR[i],alpha=0.5,label=c if c != 'pred' else 'NNET',bins=50,density=True)
    t.name = c if c != 'pred' else 'NNET'
    S.append(t)
plt.grid()
plt.ylabel(r'$R^2 per Firm$')
plt.legend()
plt.tight_layout()
plt.savefig(fig_save_dir+'hist_stock_r2.png')
plt.show()

# boxplot
t=pd.DataFrame(S).T
t.boxplot()
plt.tight_layout()
plt.savefig(fig_save_dir+'box_stock_r2.png')
plt.show()

### violin noow :)
import seaborn as sns
t1 = t[['NNET']].copy().rename(columns={'NNET':r'$R^2$ per firm'})
t1['model'] = 'NNET'
t2 = t[['vilk']].copy().rename(columns={'vilk':r'$R^2$ per firm'})
t2['model'] = 'vilk'
t3 = t1.append(t2)
t3['x'] = 1
t3 = t3.reset_index()

mkt = model.data.load_additional_crsp()
mkt=mkt.groupby('permno').mean().reset_index()
mkt['Market cap. Quantile']=pd.qcut(mkt['mkt_cap'],5,labels=False,duplicates='drop')
t3=t3.merge(mkt)

sns.violinplot(data=t3,x='Market cap. Quantile',y=r'$R^2$ per firm',hue='model',palette='muted',split=True)
plt.savefig(fig_save_dir+'SIZE_violin_stock_r2.png')
plt.show()


sns.boxplot(data=t3,x='Market cap. Quantile',y=r'$R^2$ per firm',hue='model',palette='muted')
plt.tight_layout()
plt.savefig(fig_save_dir+'SIZE_box_stock_r2.png')
plt.show()

tips = sns.load_dataset("tips")

##################
# correlation of perf
##################
plt.scatter(t['vilk'],t['NNET'],color='k',marker='+')
plt.xlabel('vilknoy firm $R^2$')
plt.ylabel('NNET firm $R^2$')
plt.grid()
plt.tight_layout()
plt.savefig(fig_save_dir+'corr_frim_r2.png')
plt.show()

df_['mse_nnet'] = (df_['pred']-df_['ret1m'])**2
df_['mse_vilk'] = (df_['vilk']-df_['ret1m'])**2
df_['ym'] = df_['date'].dt.year*100+df_['date'].dt.month

tt=df_.groupby(['ym','permno'])[['mse_nnet','mse_vilk']].mean()
plt.scatter(tt['mse_vilk'],tt['mse_nnet'],color='k',marker='+')
plt.xlabel('vilknoy firm MSE')
plt.ylabel('NNET firm MSE')
plt.grid()
plt.tight_layout()
plt.savefig(fig_save_dir+'corr_firm_month_MSE.png')
plt.show()


r2(df_)
r2(df)
