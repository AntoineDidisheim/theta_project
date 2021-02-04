import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from parameters import *
from data import *

data = Data(par=Params())
data.clean_opt_4()
