import pandas as pd
import tensorflow as tf
import numpy as np


def compute_numeric_solutions():
    pass


def sim_data(count=100):
    µ = []
    c = []
    O = []
    Px = []
    Pt = []

    for c in range(0, count):
        µ.append()
        c
        O
        Px
        Pt
        pass
        # get randomvalues for params
        # solve for dP/dx

    return {
        "µ": µ,
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












