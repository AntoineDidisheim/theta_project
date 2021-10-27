import pandas as pd
import time
from pandarallel import pandarallel
import math
import numpy as np
from parameters import *
from data import Data
from matplotlib import pyplot as plt
from ml_model import NetworkMean
par = Params()
par.data.H = 120
data=Data(par)


df = data.load_kelly()

df.head()
df['date'].unique()
df['gvkey'].unique().shape