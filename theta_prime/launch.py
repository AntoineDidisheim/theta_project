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
# par.name_detail='WithRET_'
# par.name_detail='NoMedian_'
par.name_detail='Subset_'
par.data.cs_sample = CSSAMPLE.FULL
par.model.layers = [64,32,16]
par.model.dropout = 0.2
par.model.output_range=0.2
par.model.learning_rate = 0.001
par.model.loss = Loss.MSE
par.data.H = 20
par.model.batch_normalization = False
par.model.regulator = False
par.data.var_subset = []


par.update_model_name()



data = Data(par)

trainer = Trainer(par)
self = trainer
# trainer.launch_training_expanding_window()
trainer.create_paper()

