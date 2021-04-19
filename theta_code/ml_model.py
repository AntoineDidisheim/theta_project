import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from parameters import *
from data import Data
import os
import pickle
from matplotlib import pyplot as plt
from Econ import  Econ
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

        self.save_dir = 'model_save/' + self.par.name + '/'
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def get_pred(self, x):
        pred = self.model.predict(x)
        p_log = tf.map_fn(fn=Econ.g_apply_log, elems=pred)
        p_norm = tf.map_fn(fn=Econ.g_apply_ret, elems=pred)

        return p_log, p_norm

    def get_theta(self, x):
        pred = self.model.predict(x)
        return pred[:, -1]

    def get_perf_oos(self):
        X = [self.data.test_p_df.values, self.data.test_m_df.values]
        y = self.data.test_label_df[['ret']].values
        pred = self.model.predict(X)

        p_log = tf.map_fn(fn=Econ.g_apply_log, elems=pred)
        p_norm = tf.map_fn(fn=Econ.g_apply_ret, elems=pred)
        if self.par.data.ret == ReturnType.LOG:
            p = p_log
        if self.par.data.ret == ReturnType.RET:
            p = p_norm
        theta = pred[:, -1]
        r2 = r2_score(y, p)
        mse = np.mean((p - y) ** 2)
        return r2, theta, mse, p_log, p_norm

    def get_perf_oos_normal_ret(self):
        X = [self.data.test_p_df.values, self.data.test_m_df.values]
        y = self.data.test_label_df[['normal_ret']].values
        pred = self.model.predict(X)
        p = tf.map_fn(fn=Econ.g_apply_ret, elems=pred)
        theta = pred[:, -1]
        r2 = r2_score(y, p)
        mse = np.mean((p - y) ** 2)
        return r2, theta, p, mse

    def get_perf_oos_log_ret(self):
        X = [self.data.test_p_df.values, self.data.test_m_df.values]
        y = self.data.test_label_df[['log_ret']].values
        pred = self.model.predict(X)
        p = tf.map_fn(fn=Econ.g_apply_log, elems=pred)
        theta = pred[:, -1]
        r2 = r2_score(y, p)
        mse = np.mean((p - y) ** 2)
        return r2, theta, p, mse


    def get_bench_perf(self):
        m = self.data.test_m_df.copy()
        y = self.data.test_label_df[['ret']].values
        m['theta'] = 1.0
        if self.par.data.ret == ReturnType.LOG:
            p = tf.map_fn(fn=Econ.g_apply_log, elems=m.values)
        if self.par.data.ret == ReturnType.RET:
            p = tf.map_fn(fn=Econ.g_apply_ret, elems=m.values)
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

        self.outputs = layers.Dense(1, activation="sigmoid", name='theta_forecast', dtype=tf.float64)(f)*(self.par.model.output_range-self.par.model.out_min) + self.par.model.out_min
        self.outputs = tf.concat([other_inputs, self.outputs], axis=1)
        model = keras.Model(inputs=[theta_x, other_inputs], outputs=self.outputs)

        print(model.summary())

        keras.utils.plot_model(model, "f.png", show_shapes=True, show_layer_names=True)

        if self.par.model.opti == Optimizer.ADAM:
            optimizer = tf.keras.optimizers.Adam(self.par.model.learning_rate)
        if self.par.model.opti == Optimizer.SGD:
            optimizer = tf.keras.optimizers.SGD(self.par.model.learning_rate)
        if self.par.model.opti == Optimizer.RMS_PROP:
            optimizer = tf.keras.optimizers.RMSprop(self.par.model.learning_rate)
        if self.par.model.opti == Optimizer.ADAGRAD:
            optimizer = tf.keras.optimizers.Adagrad(self.par.model.learning_rate)
        if self.par.data.ret == ReturnType.LOG:
            if self.par.model.loss == Loss.MAE:
                model.compile(loss=NetworkTheta.custom_loss_mae_log, optimizer=optimizer)
            if self.par.model.loss == Loss.MSE:
                model.compile(loss=NetworkTheta.custom_loss_mse_log, optimizer=optimizer)
        if self.par.data.ret == ReturnType.RET:
            if self.par.model.loss == Loss.MAE:
                model.compile(loss=NetworkTheta.custom_loss_mae_ret, optimizer=optimizer)
            if self.par.model.loss == Loss.MSE:
                model.compile(loss=NetworkTheta.custom_loss_mse_ret, optimizer=optimizer)
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
        # if self.model is None:
        #     self.create_nnet_model()
        self.model.load_weights(save_this_one)
        print('model loaded')



    @staticmethod
    def custom_loss_mae_log(y_true, y_pred):
        p = tf.map_fn(fn=Econ.g_apply_log, elems=y_pred)
        return tf.reduce_mean(tf.losses.MAE(y_true, p))

    @staticmethod
    def custom_loss_mse_log(y_true, y_pred):
        p = tf.map_fn(fn=Econ.g_apply_log, elems=y_pred)
        return tf.reduce_mean(tf.losses.MSE(y_true, p))

    @staticmethod
    def custom_loss_mae_ret(y_true, y_pred):
        p = tf.map_fn(fn=Econ.g_apply_ret, elems=y_pred)
        return tf.reduce_mean(tf.losses.MAE(y_true, p))

    @staticmethod
    def custom_loss_mse_ret(y_true, y_pred):
        p = tf.map_fn(fn=Econ.g_apply_ret, elems=y_pred)
        return tf.reduce_mean(tf.losses.MSE(y_true, p))



# par = Params()
# par.name_detail = 'new_version'
# par.model.tex_dir = 'new_version'
# par.model.cv = CrossValidation.YEAR_BY_YEAR
# par.model.activation = 'swish'
# par.model.learning_rate=1e-2
# par.model.layers = [10]
# par.model.batch_size = 32
# par.model.dropout = 0.0
# par.model.output_range = 1.2
# # par.model.output_range = 5.0
# par.model.E = 5
# par.data.val_split = 0.1
# par.model.loss = Loss.MAE
# par.data.opt_smooth = OptSmooth.INT
# par.data.comp = True
# par.data.ret = ReturnType.LOG
#
# self = NetworkTheta(par)
# data =Data(par)
# data.load_final()
# df = data.m_df.copy()
# df['theta'] = 0.69
# m = df.iloc[100,:].values
