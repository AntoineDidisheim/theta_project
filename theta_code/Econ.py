
import numpy as np
import tensorflow as tf
from parameters import Constant

class Econ:
    ##################
    # static function
    ##################
    @staticmethod
    def phi1(S, theta):
        return (S ** theta) * tf.math.log(S)

    @staticmethod
    def dPhi1(S, theta):
        return (S ** (theta - 1)) * (theta * tf.math.log(S) + 1)

    @staticmethod
    def ddPhi1(S, theta):
        return (S ** (theta - 2.0)) * (theta * (theta - 1) * tf.math.log(S) + 2 * theta - 1)

    @staticmethod
    def phi2(S, theta):
        return S ** theta

    @staticmethod
    def dPhi2(S, theta):
        return theta * (S ** (theta - 1))

    @staticmethod
    def ddPhi2(S, theta):
        return theta * (theta - 1) * (S ** (theta - 2))

    @staticmethod
    def tf_trapz(inp):
        return tf.numpy_function(np.trapz, inp, tf.float64)

    @staticmethod
    def g_apply_ret(m):
        # old version without numerical split
        # m = np.concatenate([K, PRICE, rf, s0])
        K = m[:Constant.GRID_SIZE]
        PRICE = m[Constant.GRID_SIZE:(2*Constant.GRID_SIZE)]
        rf = m[(2*Constant.GRID_SIZE)]
        S = m[(2*Constant.GRID_SIZE)+1]
        theta = m[(2*Constant.GRID_SIZE)+1]

        def trapezoidal_integral_approx(t, y):
            return tf.reduce_sum(tf.multiply(t[1:] - t[:-1], (y[1:] + y[:-1]) / 2.), name='trapezoidal_integral_approx')

        int_phi_up = trapezoidal_integral_approx(K, Econ.ddPhi2(K, theta + 1) * PRICE) / (1 + rf)
        int_phi_down = trapezoidal_integral_approx(K, Econ.ddPhi2(K, theta) * PRICE) / (1 + rf)

        up = Econ.phi2(S, theta + 1.0) + Econ.dPhi2(S, theta + 1.0) * S * rf + int_phi_up
        down = Econ.phi2(S, theta) + Econ.dPhi2(S, theta) * S * rf + int_phi_down

        factor = (1 / S ** (1 + theta)) / (1 / S ** theta)
        res = (up / down) * factor - 1.0
        return res

    @staticmethod
    def g_apply_log(m):
        # old version without numerical split
        # m = np.concatenate([K, PRICE, rf, s0])
        K = m[:Constant.GRID_SIZE]
        PRICE = m[Constant.GRID_SIZE:(2*Constant.GRID_SIZE)]
        rf = m[(2*Constant.GRID_SIZE)]
        S = m[(2*Constant.GRID_SIZE)+1]
        theta = m[(2*Constant.GRID_SIZE)+2]

        def trapezoidal_integral_approx(t, y):
            return tf.reduce_sum(tf.multiply(t[1:] - t[:-1], (y[1:] + y[:-1]) / 2.), name='trapezoidal_integral_approx')

        int_phi1 = trapezoidal_integral_approx(K, Econ.ddPhi1(K, theta) * PRICE) / (1 + rf)
        int_phi2 = trapezoidal_integral_approx(K, Econ.ddPhi2(K, theta) * PRICE) / (1 + rf)

        up = Econ.phi1(S, theta) + Econ.dPhi1(S, theta) * S * rf + int_phi1
        down = Econ.phi2(S, theta) + Econ.dPhi2(S, theta) * S * rf + int_phi2
        res = (up / down) - tf.math.log(S)
        return res