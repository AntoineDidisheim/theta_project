import pandas as pd
import numpy as np
import tensorflow_probability as tfp
import tensorflow as tf
from parameters import *

np.random.seed(12345)

# create a fake data
# Input to create the theta dim N*p
# Call prices N*c <- vary for each day so it's a tensor or I can't do batch
# Strike N*k
# Put prices N*p <- ibidem
# Strike N*p
# rf scalar
# fr scalar

N = 1000
max_opt = 50
rf = np.array([0.0001 for x in range(N)]).reshape(-1, 1)  # risk free t->T
fr = np.ones_like(rf)  # forward price t->T
N_c = np.random.randint(10, 50, size=(N, 1))

kc = np.random.uniform(100, 140, size=(N, max_opt))
N_p = np.random.randint(10, 50, size=(N, 1))
kp = np.random.uniform(70, 100, size=(N, max_opt))
calls = np.random.uniform(0.1, 2.0, size=(N, max_opt))
puts = np.random.uniform(0.1, 2.0, size=(N, max_opt))

X = np.random.normal(size=(N, 100))
y = np.random.normal(size=(N, 1))

data = (X, calls, puts, kc, kp, N_c, N_p, fr, rf)
data = [tf.convert_to_tensor(x) for x in data]
for x in data:
    print(x.shape)


class NetDataOutput(tf.keras.layers.Layer):
    def __init__(self, num_outputs, par):
        super(NetDataOutput, self).__init__(dtype='float64')
        self.num_outputs = num_outputs
        self.par = par

    def build(self, input_shape):
        self.kernel = []
        self.const = []
        old_l = input_shape[0][1]
        for i, l in enumerate(self.par.model.layers):
            self.kernel.append(self.add_weight("kernel_mat", shape=[old_l, l], dtype=tf.float64))
            self.const.append(self.add_weight("kernel_mat", shape=[l], dtype=tf.float64))
            old_l = l

    def call(self, input, **kwargs):
        r = input[0]
        for i, l in enumerate(self.par.model.layers):
            r = tf.matmul(r, self.kernel[i]) + self.const[i]
            if l != 1:
                r = tf.nn.swish(r)

        return [r,input[1]]


par = Params()
L = []
L.append(NetDataOutput(10, par))
model = tf.keras.Sequential(L)
optimizer = tf.keras.optimizers.Adam(par.model.learning_rate)



def custom_loss(y_true, y_pred):
    return tf.losses.MAE(y_true,y_pred[0])

model.compile(loss=custom_loss, optimizer=optimizer, metrics=['mae', 'mse'])

model.fit(data, y)
