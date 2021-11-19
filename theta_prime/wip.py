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



# df.quantile(0.99).round(2)
# df.quantile(0.01).round(2)




