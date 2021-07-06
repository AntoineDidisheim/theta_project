from data import *
from parameters import  *


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


self = Data(par)

# self.load_all_price(reload=True)
#
# if self.par.data.opt_smooth in [OptSmooth.VOLA_CUBIC]:
#     self.clean_vola_surface()
# else:
#     self.clean_opt_all()
#
# self.martin_wagner_var_mkt(reload=True)
# self.marting_wagner_return(reload=True)
# print('start', flush=True)
#
# # self.gen_all_int()
#
# if self.par.data.crsp:
#     if self.par.data.comp:
#         self.load_pred_compustat_and_crsp(reload=True)
#     else:
#         self.load_pred_crsp_only(reload=True)
#
# self.historical_theta(reload=True)

for i in range(5,10):
    self.create_a_dataset_batch(i, batch_size=25000)

