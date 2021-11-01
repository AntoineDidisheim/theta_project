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
par.name_detail='V2KellyCracking_'
par.data.cs_sample = CSSAMPLE.KELLY
par.model.layers = [32,16,8]
par.model.dropout = 0.0
par.model.output_range = 0.5
par.model.learning_rate = 0.005
par.model.output_pos_only =False
par.model.regulator=0.00001
# par.model.learning_rate = 0.001
par.model.loss = Loss.MAE
par.model.E = 5

H =20


par.data.H = H
if H>30:
    par.model.output_range = 0.5

par.update_model_name()

# data = Data(par)
# data.load_tr_kelly(True)
# data.load_feature_kelly(True)
#
# train
trainer = Trainer(par)
self = trainer
trainer.launch_training_expanding_window()
trainer.create_paper()

