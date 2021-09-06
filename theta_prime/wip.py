import pandas as pd

import time
from pandarallel import pandarallel
import math
import numpy as np
from parameters import  *
from data import Data
from matplotlib import pyplot as plt
from ml_model import NetworkMean
par = Params()
Data(par).load_pred_feature(True)

model = NetworkMean(par)

YEAR=np.sort(model.data.label_df['date'].dt.year.unique())[3:]

for year in YEAR:
    model.run_year(year)