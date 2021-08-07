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
    directory='D:/node-app/isthischae/images/processed',
    labels="inferred",
    label_mode="categorical",
    image_size=(256,256),
    # smart_resize=True,
    # batch_size=32,
    # shuffle=True,
    seed=5293056,
    validation_split=0.1,
    subset="training"
).shuffle(32, reshuffle_each_iteration=True).map(augment)

val_ds: tf.data.Dataset = tf.keras.preprocessing.image_dataset_from_directory(
    directory='D:/node-app/isthischae/images/processed',
    labels="inferred",
    label_mode="categorical",
    image_size=(256,256),
    # smart_resize=True,
    # batch_size=15,
    # shuffle=True,
    seed=5293056,
    validation_split=0.1,
    subset="validation"
)

# ds_labels = np.array([99], int)
# for images, labels in val_ds.take(1).as_numpy_iterator():
#     for label in labels:
#         l: int = np.array([np.argmax(label)])
#         ds_labels = np.concatenate((ds_labels, l), axis=0)
# ds_labels = np.delete(ds_labels, [0])
# print(len(ds_labels))
# print(ds_labels)
