import pandas as pd
import numpy as np
import tensorflow_probability as tfp
import tensorflow_probability as tf

np.random.seed(12345)

theta = 0.5

def fun(r):
    return r ** theta


def fun_der_sec(r):
    return theta * (theta - 1) * (r ** (theta - 2))


rf = 0.0001  # risk free t->T
fr = 1  # forward price t->T
N_c = 1000
kc = np.random.uniform(100, 140, N_c)
N_p = 900
kp = np.random.uniform(70, 100, N_p)

calls = np.random.uniform(0.1, 2.0, N_c)
puts = np.random.uniform(0.1, 2.0, N_p)

fin_kc = fun_der_sec(kc/fr)
fin_kp = fun_der_sec(kp/fr)

b1 = np.trapz(x=kc, y=fin_kc * calls)
b2 = np.trapz(x=kp, y=fin_kp * puts)

res = (b1 + b2) * rf / (fr ** 2) + fun(1)
print(res)

np.trapz([1,2,3],[4,5,6])