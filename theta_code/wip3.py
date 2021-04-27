from data import *
from parameters import  *


par = Params()
par.name_detail = 'new_version'
par.model.tex_dir = 'new_version'
par.model.cv = CrossValidation.YEAR_BY_YEAR
par.model.activation = 'swish'
par.model.learning_rate = 1e-2
par.model.layers = [10]
par.model.batch_size = 32
par.model.dropout = 0.0
par.model.output_range = 1.2
# par.model.output_range = 5.0
par.data.min_opt_per_day = 3
# par.data.mw  =True


par.model.E = 5
par.data.val_split = 0.1
par.model.loss = Loss.MAE
par.data.opt_smooth = OptSmooth.EXT
par.data.comp = True
par.data.ret = ReturnType.RET
data = Data(par)

opt = pd.DataFrame()
for year in range(1996,2020):
    print('year',year)
    # year=1996
    t  = pd.read_pickle(par.data.dir + f'raw/opt_{year}.p')



    ################# ===================== here
    t['exist'] = True
    t.columns = [x.lower() for x in t.columns]
    t['date'] = pd.to_datetime(t['date'])
    t['expiration'] = pd.to_datetime(t['expiration'])
    t['T'] = (t['expiration'] - t['date']).dt.days

    # t=t.loc[t['T']==28,:]
    t = t.loc[t['date'].dt.dayofweek == 4, :]
    t= t[['date','permno','exist']].drop_duplicates()
    t['permno'] = t['permno'].astype(int)
    opt=opt.append(t)

them = pd.read_csv('data/MartinWagnerBounds.csv').rename(columns={'id':'permno'})
them['date'] = pd.to_datetime(them['date'])

ind = them['date'].isin(opt['date'].unique()) & them['permno'].isin(opt['permno'].unique())
them.loc[ind,:].groupby('date')['permno'].count().plot(label='them')
opt.groupby('date')['permno'].count().plot(label='us')
plt.legend()
plt.grid()
plt.show()


them['permno'] = them['permno'].astype(int)

opt=opt.merge(them,how='left')
pd.isna(opt['mw30']).mean()
them = them.merge(opt[['date','permno','exist']], how='left')
ind = them['date'].isin(opt['date'].unique()) & them['permno'].isin(opt['permno'].unique())
ind.mean()
them.loc[ind,:]

pd.isna(opt['mw30']).mean()
print('opt.shape',opt.shape)
print('mean',pd.isna(them.loc[ind,'exist']).mean())
print('sum',pd.isna(them.loc[ind,'exist']).sum())

