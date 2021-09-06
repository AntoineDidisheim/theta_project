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

class NetworkMean:
    def __init__(self, par: Params):
        self.par = par
        self.model = None
        self.data = Data(par)
        self.create_network()

        self.save_dir = 'model_save/' + self.par.name + '/'
        self.res_dir = 'res/' + self.par.name + '/'
        os.makedirs(self.save_dir,exist_ok=True)
        os.makedirs(self.res_dir,exist_ok=True)

    def _get_perf_oos(self):
        X = self.data.test_x_df
        pred = self.model.predict(X)
        df = self.data.test_label_df.copy()
        df['pred'] = pred
        v=self.data.load_vilknoy()
        df = df.merge(v,how='left')

        def r2(df_,col='pred'):
            r2_pred = 1 - ((df_['ret1m'] - df_[col]) ** 2).sum() / ((df_['ret1m'] - 0) ** 2).sum()
            return r2_pred

        print('us overall', r2(df),'us on vilk',r2(df.dropna()), 'vilk perf',r2(df.dropna(),'vilk'))
        return df


    def shapeley_oos(self):
        d=self.data.test_label_df.copy()
        for c in self.data.test_p_df.columns:
            print('shapeley', c,flush=True)
            p = self.data.test_p_df.copy()
            p[c] = 0.0
            X = [p.values, self.data.test_m_df.values]
            pred = self.model.predict(X)
            p_norm = tf.map_fn(fn=Econ.g_apply_ret, elems=pred)
            d[c] = p_norm
        return d

    def create_network(self):

        ##################
        # main network
        ##################
        with tf.name_scope("main_network"):
            L = []
            for i, l in enumerate(self.par.model.layers):
                print(i, l)
                dense = layers.Dense(l, activation=self.par.model.activation, dtype=tf.float64)
                L.append(dense)

        def final_act(x):
            return tf.nn.tanh(x)*self.par.model.output_range
        self.outputs = layers.Dense(1, activation=final_act, name='final_forecast', dtype=tf.float64)
        L.append(self.outputs)
        model = tf.keras.Sequential(L)



        if self.par.model.opti == Optimizer.ADAM:
            optimizer = tf.keras.optimizers.Adam(self.par.model.learning_rate)
        elif self.par.model.opti == Optimizer.SGD:
            optimizer = tf.keras.optimizers.SGD(self.par.model.learning_rate)
        elif self.par.model.opti == Optimizer.RMS_PROP:
            optimizer = tf.keras.optimizers.RMSprop(self.par.model.learning_rate)
        elif self.par.model.opti == Optimizer.ADAGRAD:
            optimizer = tf.keras.optimizers.Adagrad(self.par.model.learning_rate)
        else:
            optimizer = tf.keras.optimizers.Adagrad(self.par.model.learning_rate)

        if self.par.model.loss == Loss.MAE:
            model.compile(loss='mae', optimizer=optimizer)
        if self.par.model.loss == Loss.MSE:
            model.compile(loss='mse', optimizer=optimizer)

        self.model = model

    def run_year(self,year):
        print('#'*50)
        print('###',year)
        print('#'*50,flush=True)
        self.data.set_year_test(year)
        self._train_year(year)
        df=self._get_perf_oos()
        df.to_pickle(self.res_dir+f'{year}.p')




    def _train_year(self, year):

        X = self.data.train_x_df
        y = self.data.train_label_df[['ret1m']].values
        save_this_one = self.save_dir + f'{year}/'
        if not os.path.exists(save_this_one):
            os.makedirs(save_this_one)
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=save_this_one, save_weights_only=True, verbose=0, save_best_only=True)

        self.model.fit(X, y, epochs=self.par.model.E, batch_size=self.par.model.batch_size, validation_split=0.1, callbacks=[cp_callback])
        self.load(year)

    def load(self,year):
        save_this_one = self.save_dir + f'{year}/'
        self.model.load_weights(save_this_one)
        print('model loaded')



    @staticmethod
    def custom_loss_mae_log(y_true, y_pred):
        p = tf.map_fn(fn=Econ.g_apply_log, elems=y_pred)
        return tf.reduce_mean(tf.losses.MAE(y_true, p))





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
