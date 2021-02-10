
##################
# this implement a simple/pseudo code version that takes as a cost function the fabio theta funciton
##################

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from parameters import *
from data import Data
from sklearn.metrics import r2_score

tf.random.set_seed(1234)
np.random.seed(1234)

##################
# simulated data
##################
class NetworkTheta:
    def __init__(self, par: Params):
        self.par = par
        self.model = None
        self.data = Data(par)
        self.data.load_final()
        self.create_network()

    def get_pred(self, x):
        pred = self.model.predict(x)
        up = tf.map_fn(fn=NetworkTheta.get_up, elems=pred)
        down = tf.map_fn(fn=NetworkTheta.get_down, elems=pred)
        p = up / down
        return p

    def get_theta(self, x):
        pred = self.model.predict(x)
        return pred[:,-1]


    def create_network(self):
        other_inputs = keras.Input(shape=self.data.m_df.shape[1], name='other_inputs',dtype=tf.float64)
        ##################
        # theta network
        ##################
        with tf.name_scope("theta_network"):
            theta_x = keras.Input(shape=self.data.p_df.shape[1], name='theta_input_x',dtype=tf.float64)

            f = theta_x
            for i, l in enumerate(self.par.model.layers):
                print(i, l)
                dense = layers.Dense(l, activation=self.par.model.activation,dtype=tf.float64)
                # if self.par.model.dropout >0:
                #
                f = dense(f)

        self.outputs = layers.Dense(1, activation="sigmoid", name='theta_forecast',dtype=tf.float64)(f)
        self.outputs = tf.concat([other_inputs, self.outputs], axis=1)
        model = keras.Model(inputs=[theta_x, other_inputs], outputs=self.outputs)

        print(model.summary())

        keras.utils.plot_model(model, "f.png", show_shapes=True, show_layer_names=True)

        optimizer = tf.keras.optimizers.Adam(self.par.model.learning_rate)


        # def r_square(y_true, y_pred):
        #     SS_res = tf.reduce_sum(tf.square(y_true - y_pred))
        #     SS_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
        #     return (1 - SS_res / (SS_tot + tf.keras.backend.epsilon()))

        model.compile(loss=NetworkTheta.custom_loss, optimizer=optimizer)
        self.model = model


    def train(self):
        X = [self.data.p_df.values, self.data.m_df.values]
        y = self.data.label_df[['ret']].values

        self.model.fit(X, y, epochs=self.par.model.E, batch_size=self.par.model.batch_size)

    ##################
    # static function
    ##################
    @staticmethod
    @tf.function
    def up_fun(r, theta):
        # return tf.cast((r ** theta) * tf.math.log(r),tf.float64)
        return (r ** theta) * tf.math.log(r)

    @staticmethod
    def up_fun_der_sec(r, theta):
        return  (r ** (theta - 2.0)) * (2.0*theta + (theta-1.0)*theta*tf.math.log(r)-1.0)

    @staticmethod
    def down_fun(r, theta):
        return r ** theta

    @staticmethod
    def down_fun_der_sec(r, theta):
        return tf.cast(theta * (theta - 1) * (r ** (theta - 2)),tf.float64)

    @staticmethod
    def tf_trapz(inp):
        return tf.numpy_function(np.trapz, inp, tf.float64)

    @staticmethod
    def compute_one(m, fun, fun_der_sec):
        max_opt = Constant.MAX_OPT # need to be defined here because cost function has to be static, to fix.
        # other_X = np.concatenate([kc, kp, calls, puts, rf, fr, Nc, Np], 1)
        kc = m[:max_opt]
        kp = m[max_opt:(max_opt * 2)]
        calls = m[(max_opt * 2):(max_opt * 3)]
        puts = m[(max_opt * 3):(max_opt * 4)]
        rf = m[(max_opt * 4):(max_opt * 4 + 1)]+1.0
        fr = m[(max_opt * 4 + 1):(max_opt * 4 + 2)]
        Nc = tf.cast(m[(max_opt*4+2):(max_opt*4+3)][0], tf.int64)
        Np = tf.cast(m[(max_opt*4+3):(max_opt*4+4)][0], tf.int64)
        theta = m[(max_opt * 4 + 4):(max_opt * 4 + 5)][0]
        # theta = 0.0
        calls = calls[:Nc]
        kc = kc[:Nc]
        puts = puts[:Np]
        kp = kp[:Np]

        fin_kc = fun_der_sec(kc / fr, theta)
        fin_kp = fun_der_sec(kp / fr, theta)

        def trapezoidal_integral_approx(t, y):
            return tf.reduce_sum(tf.multiply(t[1:] - t[:-1],(y[1:] + y[:-1]) / 2.), name='trapezoidal_integral_approx')
        b1 = trapezoidal_integral_approx(kc, fin_kc*calls)
        b2 = trapezoidal_integral_approx(kp, fin_kp*puts)

        res = (b1 + b2) * rf / (fr**2) + fun(tf.cast(1.0,tf.float64), theta)
        return res

    @staticmethod
    def get_up(m):
        return NetworkTheta.compute_one(m, fun=NetworkTheta.up_fun, fun_der_sec=NetworkTheta.up_fun_der_sec)

    @staticmethod
    def get_down(m):
        return NetworkTheta.compute_one(m, fun=NetworkTheta.down_fun, fun_der_sec=NetworkTheta.down_fun_der_sec)

    @staticmethod
    def custom_loss(y_true, y_pred):
        up = tf.map_fn(fn=NetworkTheta.get_up, elems=y_pred)
        down = tf.map_fn(fn=NetworkTheta.get_down, elems=y_pred)
        p = up/down
        return tf.reduce_mean(tf.losses.MSE(y_true, p))

    @staticmethod
    def custom_debug(y_true, y_pred):

        up = tf.map_fn(fn=NetworkTheta.get_up, elems=y_pred)
        down = tf.map_fn(fn=NetworkTheta.get_down, elems=y_pred)
        p = up/down
        # return tf.reduce_mean(tf.square(p-y_true))
        # p = y_pred[:,-1]
        return p




# # debug
par = Params()
par.model.layers = [64,32,16]
# par.model.layers = [10]
par.model.activation = 'sigmoid'
par.model.batch_size = 32
par.model.learning_rate = 0.01
par.model.E = 5
self = NetworkTheta(par)
# #

X = [self.data.p_df.values, self.data.m_df.values]
y = self.data.label_df[['ret']].values
X_test = [self.data.p_test.values, self.data.m_test.values]
y_test = self.data.label_test[['ret']].values



theta_before = self.get_theta(X_test)
pred_os_before=self.get_pred(X_test)
pred_is_before=self.get_pred(X)
print('before train r2 os', r2_score(y_test, pred_os_before))
print('before train r2 is', r2_score(y, pred_is_before))
self.train()
pred_os_after=self.get_pred(X_test)
pred_is_after=self.get_pred(X)
theta_after = self.get_theta(X_test)
print('after train r2 os', r2_score(y_test, pred_os_after))
print('after train r2 is', r2_score(y, pred_is_after))



print('theta unchanged:', all(theta_after == theta_before))


##################
# to beat
##################
print('###################')
print('benchmark')
print('###################')
m = self.data.m_df.copy()
m['theta'] = 1.0
pred_1 = NetworkTheta.custom_debug(y, m.values)
m['theta'] = 0.0
pred_0 = NetworkTheta.custom_debug(y, m.values)


print('is, r2 0.0', r2_score(y, pred_0))
print('is, r2 1.0', r2_score(y, pred_1))

m = self.data.m_test.copy()
m['theta'] = 1.0
pred_1_test = NetworkTheta.custom_debug(y_test, m.values)
m['theta'] = 0.0
pred_0_test = NetworkTheta.custom_debug(y_test, m.values)
m['theta'] = 0.5
pred_05_test = NetworkTheta.custom_debug(y_test, m.values)


print('os, r2 0.0', r2_score(y_test, pred_0_test))
print('os, r2 1.0', r2_score(y_test, pred_1_test))


y_all = np.concatenate([y_test,y])
us_all_after = np.concatenate([pred_os_after,pred_is_after])
us_all_before = np.concatenate([pred_os_before,pred_is_before])
pred_all_1 = np.concatenate([pred_1,pred_1_test])
pred_all_0 = np.concatenate([pred_0,pred_0_test])


print('ALL, r2 0.0', r2_score(y_all, pred_all_0))
print('ALL, r2 1.0', r2_score(y_all, pred_all_1))
print('ALL, r2 random_net net', r2_score(y_all, us_all_before))
print('ALL, r2 trained net', r2_score(y_all, us_all_after))

