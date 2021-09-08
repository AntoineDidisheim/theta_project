import numpy as np
import pandas as pd
from parameters import *
from trainer import Trainer
from data import  Data
import socket

if 'nv-' in socket.gethostname():
    import matplotlib
    matplotlib.use('Agg')

par = Params()
par.data.cs_sample = CSSAMPLE.VILK
# par.model.layers = [100,100,100]
par.model.dropout = 0.2
par.model.learning_rate = 0.005
data = Data(par)
# data.load_all_price(True)
# data.load_pred_feature(True)

# train
trainer = Trainer(par)
self = trainer
trainer.launch_training_expanding_window()
trainer.create_paper()
