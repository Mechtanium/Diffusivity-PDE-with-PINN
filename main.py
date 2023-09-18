import pandas as pd
import tensorflow as tf
import numpy as np
import random as rd


def compute_numeric_solutions(u, w, o, t):
    pass


def sim_data(count=100):
    y = []
    c = []
    O = []
    Px = []
    Pt = []

    for _ in range(0, count):
        u = rd.randint(1, 2) / 100
        w = rd.randint(1, 2) / 100
        o = rd.randint(1, 2) / 100
        t = rd.randint(1, 2) / 100

        y.append(u)
        c.append(w)
        O.append(o)
        Pt.append(t)
        Px.append(compute_numeric_solutions(u, w, o, t))

    return {
        "µ": y,
        "c": c,
        "∅": O,
        "∂P/∂x": Px,
        "∂P/∂t": Pt
    }


data = pd.DataFrame(sim_data(10000))

model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=4),
    tf.keras.layers.Dense(units=10000),
    tf.keras.layers.Dense(units=10000),
    tf.keras.layers.Dense(units=10000),
    tf.keras.layers.Dense(units=10000),
    tf.keras.layers.Dense(units=10000),
    tf.keras.layers.Dense(units=10000),
    tf.keras.layers.Dense(units=10000),
    tf.keras.layers.Dense(units=10000),
    tf.keras.layers.Dense(units=1)
])

opt = tf.keras.optimizers.SGD()

for i in range(0, 100):
    pred = model(data.drop("∂P/∂x", axis=1))

    with tf.GradientTape() as grad:
        loss = tf.keras.losses.MSE(data["∂P/∂x"], pred)

    gradients = grad.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(gradients, model.trainable_variables))

print(loss)
