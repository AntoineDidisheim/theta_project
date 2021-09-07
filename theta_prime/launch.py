import numpy as np
import pandas as pd
from parameters import *
from trainer import Trainer
from data import  Data
par = Params()
par.data.cs_sample = CSSAMPLE.VILK
# par.model.layers = [100,100,100]

data = Data(par)
# data.load_all_price(True)
# data.load_pred_feature(True)

# train
trainer = Trainer(par)
trainer.launch_training_expanding_window()
trainer.create_paper()
