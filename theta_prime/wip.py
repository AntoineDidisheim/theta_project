import pandas as pd
import time
from pandarallel import pandarallel
import math
import numpy as np
from parameters import *
from data import Data
from matplotlib import pyplot as plt
from ml_model import NetworkMean
import sqlite3
par = Params()
par.data.H = 20
data=Data(par)


df = data.load_pred_feature(reload=False)
print(df.head())
print(df.columns)

