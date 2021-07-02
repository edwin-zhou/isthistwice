import tensorflow as tf
import tensorflowjs as tfjs
import numpy as np
import matplotlib.pyplot as plt
import math
import model as M
import sys

# model: tf.keras.Sequential = v4.create_model(num_classes=9)

model: tf.keras.Sequential = M.model

model: tf.keras.Sequential = tf.keras.models.load_model("../models/test1/keras/model.h5")
model.compile(optimizer=tf.optimizers.Adam(1e-5), loss=tf.keras.losses.categorical_crossentropy, metrics=tf.keras.metrics.categorical_accuracy)

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
    seed=529056,
    validation_split=0.1,
    subset="training"
).prefetch(tf.data.AUTOTUNE).shuffle(64, reshuffle_each_iteration=True).map(augment)

val_ds: tf.data.Dataset = tf.keras.preprocessing.image_dataset_from_directory(
    directory='D:/node-app/isthischae/images/processed',
    labels="inferred",
    label_mode="categorical",
    image_size=(256,256),
    # smart_resize=True,
    # batch_size=15,
    # shuffle=True,
    seed=529056,
    validation_split=0.1,
    subset="validation"
)

def plot(dset: tf.data.Dataset=train_ds, num=1):
    for e, l in dset.take(num):
        plt.figure(figsize=(10, 10))
        for i in range(len(e)):
            ax = plt.subplot(math.ceil(math.sqrt(len(e))),math.ceil(math.sqrt(len(e))),i+1)
            plt.imshow(e[i].numpy()/255)
            plt.title(l[i].numpy().nonzero()[0].astype(str))
            plt.axis("off")
        plt.show()

def train(m: tf.keras.Sequential = model):
    history = m.fit(
        x=train_ds,
        epochs=10,
        verbose=1,
        validation_data=val_ds,
        callbacks= [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.002, patience=3, mode="auto", restore_best_weights=True)
        ]
    )
    # summarize history for accuracy
    print(history.history)
    plt.plot(history.history['categorical_accuracy'])
    plt.plot(history.history['val_categorical_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('categorical_accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    return

def pr(t: tf.Tensor, l):
    print(l.shape)
    return (t, l)

# plot(train_ds, 2)
print(model._get_compile_args())
train(model)

# tfjs.converters.save_keras_model(model, "../models/ot9")

model.save("../models/test1/keras/model.h5", include_optimizer=False)



