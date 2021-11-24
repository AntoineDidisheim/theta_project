import numpy as np
import pandas as pd
from parameters import *
from trainer import Trainer
from data import  Data
import socket
import pandas as pd
import time
from pandarallel import pandarallel
import math
import numpy as np
from parameters import  *
from data import Data
from matplotlib import pyplot as plt
from ml_model import NetworkMean
import didipack as didi
import os
from tqdm import tqdm
import seaborn as sns
from scipy.stats import pearsonr
import shutil


if 'nv-' in socket.gethostname():
    import matplotlib
    matplotlib.use('Agg')

par = Params()
par.name_detail='PostVac'
par.data.cs_sample = CSSAMPLE.FULL
par.model.layers = [100,100,100]
par.model.dropout = 0.2
par.model.output_range=0.5
par.model.learning_rate = 0.001
par.model.loss = Loss.MSE
par.data.H = 20
par.model.batch_normalization = False
par.model.regulator = False

par.update_model_name()
name = 'PostVacL100_100_100_Lr0001Dropout02BS512ActreluOutRange05pLossMSECssampleFULL'
print(par.name == name)
print(par.name)
print(name)


data = Data(par)
# data.load_all_price(True)
# data.load_pred_feature(True)

# train
trainer = Trainer(par)
self = trainer
trainer.launch_training_expanding_window()
trainer.create_paper()

