import pandas as pd
import tensorflow as tf
import numpy as np


def sim_data(count=100):
    return {
        "µ": [],
        "c": [],
        "∅": [],
        "∂P/∂x": [],
        "∂P/∂t": []
    }


data = pd.DataFrame(sim_data(10000))

model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=5),
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
y = 0
predicted = 0

mse_loss = tf.keras.losses.MSE(data["∂P/∂x"], data[""])
