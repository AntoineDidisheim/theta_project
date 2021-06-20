from data import *
from parameters import  *
from multiprocessing import Pool
import sys

try:
    grid_id = int(sys.argv[1])
except:
    print('Debug mode on local machine')
    grid_id = 0


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
par.data.ret = ReturnType.RET
par.data.min_opt_per_day = 2
par.data.mw =True
par.update_model_name()


self = Data(par)
self.historical_theta(create_batch=True)

for grid_id in range(0,27):
    print('########################', 'start working on batch', grid_id, flush=True)
    self.do_batch_hist_theta(grid_id)




# if grid_id == -1:
#     self.historical_theta(create_batch=True)
# else:
#     # 26
#     self.do_batch_hist_theta(grid_id)
