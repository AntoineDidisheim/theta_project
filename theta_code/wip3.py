from data import *
from parameters import  *
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os

d='res/opt_only64L64_32_16_Lr001Dropout00BS32ActswishOutRange50CVYEAR_BY_YEARLossMAERetRETd2OptCompCrspVOLA_CUBIC/'
os.listdir(d)
df = pd.read_pickle(d+'df_sh.p')
t = pd.read_pickle(d+'df.p')
df['year'] = df['date'].dt.year
df.groupby('year')['theta'].std().round(3)


df['theta']
