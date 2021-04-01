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
par.model.E = 5
par.data.val_split = 0.1
par.model.loss = Loss.MAE
par.data.opt_smooth = OptSmooth.EXT
par.data.comp = True
par.data.ret = ReturnType.LOG
self = Data(par)
# self.create_a_dataset
# ()
r = self.create_good_iv()
pd.Series(r).to_pickle('temp.p')