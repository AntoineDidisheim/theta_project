import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from parameters import *
from data import *


def trapezoidal_integral_approx(t, y):
    return tf.reduce_sum(
            tf.multiply(t[1:] - t[:-1],
                              (y[1:] + y[:-1]) / 2.),
            name='trapezoidal_integral_approx')


t = np.array([1,2,3,4])
y = np.array([8,4,6,7])

np.trapz(y,t)
trapezoidal_integral_approx(t,y)