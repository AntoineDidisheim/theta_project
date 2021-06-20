from data import *
from parameters import  *
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

N=100
p = 0.65
w = 2
def u(x):
    return np.log(x+w)

u_trade=(p*u(1)+(1-p)*u(-1))
u_no_trade = u(0)

u_trade-u_no_trade