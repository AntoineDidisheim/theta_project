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
import matplotlib
import pylab


if 'nv-' in socket.gethostname():
    matplotlib.use('Agg')

params = {'axes.labelsize': 14,
          'axes.labelweight': 'normal',
          'xtick.labelsize': 12,
          'ytick.labelsize': 12,
          'axes.titlesize': 12}
pylab.rcParams.update(params)

par = Params()
par.name_detail='PostVac'
par.data.cs_sample = CSSAMPLE.FULL
# par.model.layers = [100,100,100]
par.model.layers = [32,16,8]
par.model.dropout = 0.0
par.model.batch_size = 256
par.model.output_range = 0.5
par.model.learning_rate = 0.001
par.model.loss = Loss.MSE
par.model.E = 10
par.data.H = 20

name = 'PostVacL100_100_100_Lr0001Dropout02BS512ActreluOutRange05pLossMSECssampleFULL'
par.name =  name


trainer = Trainer(par)
self = trainer
trainer.create_paper()

