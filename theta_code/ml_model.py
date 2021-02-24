import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from parameters import *
from data import Data
from sklearn.metrics import r2_score
import os
import pickle
from matplotlib import pyplot as plt
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

        self.save_dir = 'model_save/' + self.par.name + '/'
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def get_pred(self, x):
        pred = self.model.predict(x)
        p = tf.map_fn(fn=NetworkTheta.g_apply, elems=pred)
        return p

    def get_theta(self, x):
        pred = self.model.predict(x)
        return pred[:, -1]

    def get_perf_oos(self):
        X = [self.data.test_p_df.values, self.data.test_m_df.values]
        y = self.data.test_label_df[['ret']].values
        pred = self.model.predict(X)
        p = tf.map_fn(fn=NetworkTheta.g_apply, elems=pred)
        theta = pred[:, -1]
        r2 = r2_score(y, p)
        mse = np.mean((p - y) ** 2)
        return r2, theta, p, mse

    def get_bench_perf(self):
        m = self.data.test_m_df.copy()
        y = self.data.test_label_df[['ret']].values
        m['theta'] = 1.0
        p = tf.map_fn(fn=NetworkTheta.g_apply, elems=m.values)
        r2 = r2_score(y, p)
        mse = np.mean((p - y) ** 2)

        return p, r2, mse

    def create_network(self):
        other_inputs = keras.Input(shape=self.data.m_df.shape[1], name='other_inputs', dtype=tf.float64)
        ##################
        # theta network
        ##################
        with tf.name_scope("theta_network"):
            theta_x = keras.Input(shape=self.data.p_df.shape[1], name='theta_input_x', dtype=tf.float64)

            f = theta_x
            for i, l in enumerate(self.par.model.layers):
                print(i, l)
                dense = layers.Dense(l, activation=self.par.model.activation, dtype=tf.float64)
                # if self.par.model.dropout >0:
                #
                f = dense(f)

        self.outputs = layers.Dense(1, activation="sigmoid", name='theta_forecast', dtype=tf.float64)(f)*self.par.model.output_range
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
        X = [self.data.train_p_df.values, self.data.train_m_df.values]
        y = self.data.train_label_df[['ret']].values
        save_this_one = self.save_dir + f'{self.data.shuffle_id}/'
        if not os.path.exists(save_this_one):
            os.makedirs(save_this_one)
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=save_this_one, save_weights_only=True, verbose=0, save_best_only=True)

        self.model.fit(X, y, epochs=self.par.model.E, batch_size=self.par.model.batch_size, validation_split=0.1, callbacks=[cp_callback])
        self.load()

    def load(self):
        save_this_one = self.save_dir + f'{self.data.shuffle_id}/'
        if self.model is None:
            self.create_nnet_model()
        self.model.load_weights(save_this_one)
        print('model loaded')

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
    def g_apply(m):

        # old version without numerical split
        # m = np.concatenate([K, PRICE, rf, s0])
        K = m[:200]
        PRICE = m[200:400]
        rf = m[400]
        S = m[401]
        theta = m[402]

        def trapezoidal_integral_approx(t,y):
            return tf.reduce_sum(tf.multiply(t[1:] - t[:-1], (y[1:] + y[:-1]) / 2.), name='trapezoidal_integral_approx')

        int_phi1 = trapezoidal_integral_approx(K, NetworkTheta.ddPhi1(K, theta) * PRICE)
        int_phi2 = trapezoidal_integral_approx(K, NetworkTheta.ddPhi2(K, theta) * PRICE)

        up = NetworkTheta.phi1(S, theta) + NetworkTheta.dPhi1(S, theta) *S* rf + int_phi1
        down = NetworkTheta.phi2(S, theta) + NetworkTheta.dPhi2(S, theta) *S* rf + int_phi2
        res = (up / down) - tf.math.log(S)

        # v split

        # m = np.concatenate([K, PRICE, rf, s0])
        # K = m[:200]
        # PRICE = m[200:400]
        # rf = m[400]
        # S = m[401]
        # theta = m[402]
        # c_ind = K<=S
        # p_ind = K >= S
        # kc = K[c_ind]
        # kp = K[p_ind]
        # price_call = PRICE[c_ind]
        # price_put = PRICE[p_ind]
        #
        # def trapezoidal_integral_approx(t, y):
        #     return tf.reduce_sum(tf.multiply(t[1:] - t[:-1], (y[1:] + y[:-1]) / 2.), name='trapezoidal_integral_approx')
        #
        # int_phi1_call = trapezoidal_integral_approx(kc, NetworkTheta.ddPhi1(kc, theta) * price_call)
        # int_phi1_put = trapezoidal_integral_approx(kp, NetworkTheta.ddPhi1(kp, theta) * price_put)
        # int_phi2_call = trapezoidal_integral_approx(kc, NetworkTheta.ddPhi2(kc, theta) * price_call)
        # int_phi2_put = trapezoidal_integral_approx(kp, NetworkTheta.ddPhi2(kp, theta) * price_put)
        #
        # up = NetworkTheta.phi1(S, theta) + NetworkTheta.dPhi1(S, theta) * S * rf + int_phi1_call+int_phi1_put
        # down = NetworkTheta.phi2(S, theta) + NetworkTheta.dPhi2(S, theta) * S * rf + int_phi2_call+int_phi2_put
        # res = (up / down) - tf.math.log(S)
        return res

    @staticmethod
    def custom_loss(y_true, y_pred):
        # p = tf.map_fn(fn=NetworkTheta.g_apply(), elems=y_pred)
        p = tf.map_fn(fn=NetworkTheta.g_apply, elems=y_pred)

        return tf.reduce_mean(tf.losses.MAE(y_true, p))

    @staticmethod
    def custom_debug(y_true, y_pred):
        p = tf.map_fn(fn=NetworkTheta.g_apply, elems=y_pred)
        return p


# # # # debug
# par = Params()
# par.model.layers = [10]
# par.model.activation = 'sigmoid'
# par.model.batch_size = 32
# par.model.learning_rate = 0.01
# par.model.E = 5
# self = NetworkTheta(par)
# #
# self.data.move_shuffle()
# m = self.data.test_m_df
# m['theta'] = 0.0
# m = m.values[0,:]
# # #
# pred=NetworkTheta.custom_debug(0,m.values)
# df=self.data.test_label_df.copy()
# df['pred'] = pred.numpy()
#
#
# X = [self.data.train_p_df.values, self.data.train_m_df.values]
# y = self.data.train_label_df[['ret']].values
# X_test = [self.data.test_p_df.values, self.data.test_m_df.values]
# y_test = self.data.test_label_df[['ret']].values
#
# theta_before = self.get_theta(X_test)
# pred_os_before=self.get_pred(X_test)
# pred_is_before=self.get_pred(X)
# print('before train r2 os', r2_score(y_test, pred_os_before))
# print('before train r2 is', r2_score(y, pred_is_before))
# self.train()
# pred_os_after=self.get_pred(X_test)
# pred_is_after=self.get_pred(X)
# theta_after = self.get_theta(X_test)
# print('after train r2 os', r2_score(y_test, pred_os_after))
# print('after train r2 is', r2_score(y, pred_is_after))
# #
#
#
# print('theta unchanged:', all(theta_after == theta_before))
#
#
# ##################
# # to beat
# ##################
# print('###################')
# print('benchmark')
# print('###################')
# m = self.data.m_df.copy()
# m['theta'] = 1.0
# pred_1 = NetworkTheta.custom_debug(y, m.values)
# m['theta'] = 0.0
# pred_0 = NetworkTheta.custom_debug(y, m.values)
#
#
# print('is, r2 0.0', r2_score(y, pred_0))
# print('is, r2 1.0', r2_score(y, pred_1))
#
# m = self.data.m_test.copy()
# m['theta'] = 1.0
# pred_1_test = NetworkTheta.custom_debug(y_test, m.values)
# m['theta'] = 0.0
# pred_0_test = NetworkTheta.custom_debug(y_test, m.values)
# m['theta'] = 0.5
# pred_05_test = NetworkTheta.custom_debug(y_test, m.values)
#
#
# print('os, r2 0.0', r2_score(y_test, pred_0_test))
# print('os, r2 1.0', r2_score(y_test, pred_1_test))
#
#
# y_all = np.concatenate([y_test,y])
# us_all_after = np.concatenate([pred_os_after,pred_is_after])
# us_all_before = np.concatenate([pred_os_before,pred_is_before])
# pred_all_1 = np.concatenate([pred_1,pred_1_test])
# pred_all_0 = np.concatenate([pred_0,pred_0_test])
#
#
# print('ALL, r2 0.0', r2_score(y_all, pred_all_0))
# print('ALL, r2 1.0', r2_score(y_all, pred_all_1))
# print('ALL, r2 random_net net', r2_score(y_all, us_all_before))
# print('ALL, r2 trained net', r2_score(y_all, us_all_after))
#
