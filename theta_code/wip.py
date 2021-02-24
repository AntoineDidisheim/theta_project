
##################
# this implement a simple/pseudo code version that takes as a cost function the fabio theta funciton
##################

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from parameters import *
from ml_model import NetworkTheta
from data import *


df = pd.read_pickle('opt.p')
df['TICKER'].unique().shape
df['DATE']=pd.to_datetime(df['DATE'])
df['m'] = df['DATE'].dt.month+df['DATE'].dt.year*100
df[['TICKER','m']].drop_duplicates().shape
df['DATE'].min()

df.shape