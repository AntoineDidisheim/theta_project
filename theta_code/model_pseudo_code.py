
##################
# this implement a simple/pseudo code version that takes as a cost function the fabio theta funciton
##################

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from parameters import *

par = Params()
par.model.layers = [400 for x in range(7)]
x_dim = 10

##################
# simulated data
##################

N = 100
max_opt = 50
rf = np.array([0.0001 for x in range(N)]).reshape(-1, 1)  # risk free t->T
fr = np.ones_like(rf)  # forward price t->T
Nc = np.random.randint(10, 50, size=(N, 1))

kc = np.random.uniform(100, 140, size=(N, max_opt))
Np = np.random.randint(10, 50, size=(N, 1))
kp = np.random.uniform(70, 100, size=(N, max_opt))
calls = np.random.uniform(0.1, 2.0, size=(N, max_opt))
puts = np.random.uniform(0.1, 2.0, size=(N, max_opt))

other_X = np.concatenate([kc, kp, calls, puts, rf, fr, Nc, Np], 1)
other_dim = other_X.shape[1]

X = np.random.normal(size=(N, x_dim))
y = X[:, 0]
y = np.ones_like(y)

other_inputs = keras.Input(shape=other_dim, name='other_inputs')

##################
# theta network
##################
with tf.name_scope("theta_network"):
    theta_x = keras.Input(shape=x_dim, name='theta_input_x')

    f = theta_x
    for i, l in enumerate(par.model.layers):
        print(i, l)
        dense = layers.Dense(l, activation=par.model.activation)
        f = dense(f)

outputs = layers.Dense(1, activation="sigmoid", name='theta_forecast')(f)


##################
# other input
##################

def fun(r, theta):
    return r ** theta


def fun_der_sec(r, theta):
    return theta * (theta - 1) * (r ** (theta - 2))



@tf.function
def tf_trapz(inp):
    return tf.numpy_function(np.trapz, inp, tf.float32)


@tf.function
def compute_one(m):
    # other_X = np.concatenate([kc, kp, calls, puts, rf, fr, Nc, Np], 1)
    kc = m[:max_opt]
    kp = m[max_opt:(max_opt * 2)]
    calls = m[(max_opt * 2):(max_opt * 3)]
    puts = m[(max_opt * 3):(max_opt * 4)]
    rf = m[(max_opt * 4):(max_opt * 4 + 1)]
    fr = m[(max_opt * 4 + 1):(max_opt * 4 + 2)]
    Nc = tf.cast(m[(max_opt*4+2):(max_opt*4+3)][0], tf.int64)
    Np = tf.cast(m[(max_opt*4+3):(max_opt*4+4)][0], tf.int64)
    theta = m[(max_opt * 4 + 4):(max_opt * 4 + 5)][0]
    calls = calls[:Nc]
    kc = kc[:Nc]
    puts = puts[:Np]
    kp = kp[:Np]

    fin_kc = fun_der_sec(kc / fr, theta)
    fin_kp = fun_der_sec(kp / fr, theta)

    b1 = tf_trapz([fin_kc * calls, kc])
    b2 = tf_trapz([fin_kp * puts, puts])

    res = (b1 + b2) * rf / (fr ** 2) + fun(1, theta)
    return res


outputs = tf.concat([other_inputs, outputs], axis=1)


def custom_loss(y_true, y_pred):
    print(y_pred.shape)
    # p = tf.vectorized_map(fn=compute_one, elems=y_pred)
    p = tf.map_fn(fn=compute_one, elems=y_pred)
    return tf.losses.MSE(y_true, p)


model = keras.Model(inputs=[theta_x, other_inputs], outputs=outputs)

print(model.summary())

keras.utils.plot_model(model, "f.png", show_shapes=True, show_layer_names=True)

optimizer = tf.keras.optimizers.Adam(par.model.learning_rate)
model.compile(loss=custom_loss, optimizer=optimizer)
model.fit([X, other_X], y, epochs=10)
