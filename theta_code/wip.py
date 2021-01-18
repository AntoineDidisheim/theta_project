import pandas as pd
import numpy as np
import tensorflow_probability as tfp
import tensorflow_probability as tf

theta = 0.5


def fun(r):
    return r ** theta


def fun_der_sec(r):
    return theta * (r ** (theta - 1))

fr = 1
N_c = 1000
kc = np.random.uniform(100, 140, N_c)
N_p = 900
kp = np.random.uniform(70, 100, N_p)

calls = np.random.uniform(0.1, 2.0, N_c)
puts = np.random.uniform(0.1, 2.0, N_p)

fin_kc = np.ones_like(calls)
fin_kp = np.ones_like(puts)

for i in range(len(kc)):
    fin_kc[i] = fun_der_sec(kc[i]/fr)