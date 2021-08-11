import tensorflow as tf
import numpy as np
import statistics

import settings

model: tf.keras.Sequential = tf.keras.models.load_model("../models/" +  settings.dir + "/keras/model.h5")

# model = tf.keras.Sequential([
#     tf.keras.layers.BatchNormalization(input_shape=(256,256,3)),
#     tf.keras.layers.Conv2D(
#         # input_shape=[256,256,3],
#         padding = "same",
#         filters=64,
#         kernel_size=5,
#         strides=1,
#         activation=tf.keras.activations.relu,
#     ),
#     tf.keras.layers.Conv2D(
#         filters=128,
#         kernel_size=3,
#         strides=1,
#         activation=tf.keras.activations.relu,
#     ),
#     tf.keras.layers.MaxPool2D(
#         pool_size=5
#     ),
#     tf.keras.layers.Conv2D(
#         filters=256,
#         kernel_size=3,
#         strides=1,
#         activation=tf.keras.activations.relu,
#     ),
#         tf.keras.layers.Conv2D(
#         filters=64,
#         kernel_size=3,
#         strides=1,
#         activation=tf.keras.activations.relu,
#     ),
#     tf.keras.layers.MaxPool2D(
#         pool_size=2
#     ),
#         tf.keras.layers.Conv2D(
#         filters=9,
#         kernel_size=1,
#         strides=1,
#         activation=tf.keras.activations.relu,
#     ),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(
#         units=36,
#         activation=tf.keras.activations.relu),
#     tf.keras.layers.Dense(
#         units=9,
#         activation=tf.keras.activations.softmax,
#     )
# ])


model.compile(
    optimizer=tf.optimizers.Adam(1e-4),
    loss=tf.keras.losses.categorical_crossentropy,
    metrics=tf.keras.metrics.categorical_accuracy)

model.summary()