import tensorflow as tf
import tensorflowjs as tfjs
import numpy as np
import matplotlib.pyplot as plt

def augment(t, label):
    r = tf.random.uniform((), 0, 4, tf.int32)
    t = tf.image.random_flip_left_right(t)
    t = tf.image.rot90(t, r)
    return (t, label)

train_ds: tf.data.Dataset = tf.keras.preprocessing.image_dataset_from_directory(
    directory='../images/processed',
    labels="inferred",
    label_mode="categorical",
    image_size=(256,256),
    batch_size=36,
    seed=5293056,
    validation_split=0.1,
    subset="training"
).map(augment).shuffle(36, reshuffle_each_iteration=False)
# 

val_ds: tf.data.Dataset = tf.keras.preprocessing.image_dataset_from_directory(
    directory='../images/processed',
    labels="inferred",
    label_mode="categorical",
    image_size=(256,256),
    seed=5293056,
    validation_split=0.1,
    subset="validation"
)