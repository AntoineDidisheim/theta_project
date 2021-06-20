import pandas as pd
import numpy as np
from scipy import optimize

class Agents:
    def __init__(self):
        self.rf = 0
        self.p = 0.5
        self.alpha = 2.0
        self.w = 1
        self.inv_g = 1.0
        self.inv_b = 1.0

    def u(self,x):
        if self.alpha == 0:
            return x
        elif self.alpha == 1:
            return np.log(x)
        else:
            return np.exp(-self.alpha*x)

    def exp_u(self,g,b,qg,qb):
        return self.p*self.u(self.w - (g*(qg-self.inv_g)+b*(qb-self.inv_b)) + qg)+\
               (1-self.p)*self.u(self.w - (g*(qg-self.inv_g)+b*(qb-self.inv_b)) + qb)

    def get_q(self,g,b):
        def func(x):
            return -self.exp_u(g,b,x[0],x[1])
        x = np.array([0.0,0.0])
        func(x)
        r=optimize.minimize(func,x)
        return r.x

a1 = Agents()
a2 = Agents()
a3 = Agents()

self = a1
g=0.4;b=0.5;
def clearing_func(x):
    q1=a1.get_q(x[0],x[1])
    q2=a2.get_q(x[0],x[1])
    q3=a3.get_q(x[0],x[1])
    q=q1+q2+q3
    return np.mean((q-np.ones_like(q))**2)

x_init = np.array([0.5,0.51])
optimize.minimize(clearing_func,x_init,bounds=[[0,1],[0,1]])