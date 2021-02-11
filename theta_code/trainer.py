import pandas as pd
import numpy as np
from parameters import *
from ml_model import *
from data import *
import os

##################
# save dir
##################
save_dir = 'res/cv/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

##################
# Set parameters
##################
par = Params()
# par.model.layers = [64,32,16]
par.model.layers = []
par.model.activation = 'sigmoid'
par.model.batch_size = 32
par.model.learning_rate = 0.001
par.model.E = 10
par.data.val_split = 0.1
res = []
par.update_model_name()


model = NetworkTheta(par)
for i in range(10):
    model.data.move_shuffle()
    r2, theta, p, mse=model.get_perf_oos()
    model.train()
    r2_new, theta_new, p_new,mse_new=model.get_perf_oos()
    p_bench, r2_bench, mse_bench =model.get_bench_perf()

    r = model.data.test_label_df.copy()
    r['pred_no_train'] = p.numpy()
    r['pred'] = p_new.numpy()
    r['bench'] = p_bench.numpy()
    r['theta'] = theta_new
    res.append(r)
    print('########### r2')
    print('old',r2,'new',r2_new,'bench',r2_bench)
    print('########### mse')
    print('old',mse,'new',mse_new,'bench',mse_bench)
    model.create_network()
df = pd.concat(res)

def r2(y_true,y_pred):
    r2 = 1-((y_true - y_pred) ** 2).sum() / ((y_true - y_true.mean()) ** 2).sum()
    return r2

r2(df['ret'],df['pred'])
r2(df['ret'],df['pred_no_train'])
r2(df['ret'],df['bench'])

((df['ret']-df['bench'])**2).mean()
((df['ret']-df['pred'])**2).mean()
((df['ret']-df['pred_no_train'])**2).mean()


