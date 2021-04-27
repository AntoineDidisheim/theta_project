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
par.model.E = 5
par.data.val_split = 0.1
par.model.loss = Loss.MAE
par.data.opt_smooth = OptSmooth.EXT
par.data.comp = True
par.data.ret = ReturnType.RET
data = Data(par)

mw =data.marting_wagner_return()
pr=data.load_all_price()[['permno','gvkey']]
pr['permno'] = pr['permno'].astype(int)
pr = pr.drop_duplicates()
mw=mw.merge(pr,how='left')

them = pd.read_csv('data/MartinWagnerBounds.csv').rename(columns={'id':'permno'})
them['date'] = pd.to_datetime(them['date'])
them['permno'] = them['permno'].astype(int)

mw=mw.merge(them,how='left')
pd.isna(mw['mw30']).mean()
them = them.merge(mw[['date','permno','MW']], how='left')
ind = them['date'].isin(mw['date'].unique()) & them['permno'].isin(mw['permno'].unique())
ind.mean()
them.loc[ind,:]
pd.isna(them.loc[ind,'MW']).mean()
pd.isna(mw['mw30']).mean()

mw['mw_yearly']=(1+mw['MW'])**(252/20)-1
mw['mw_yearly']=(mw['MW'])*252/20

y=np.linspace(0,mw['mw30'].max(),100)
x=np.linspace(0,mw['mw_yearly'].max(),100)
plt.scatter(mw['mw_yearly'],mw['mw30'],color='k',marker='+',alpha=0.8)
plt.xlim(0,2.0)
plt.ylim(0,2.0)
plt.plot(x,x,color='r')
plt.show()
