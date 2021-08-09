import tensorflow as tf
import tensorflowjs as tfjs
import numpy as np
import matplotlib.pyplot as plt
import math
import sys

import settings as s
import confusion as c
import data as d
from model import model as model

def plot(dset: tf.data.Dataset=d.train_ds, num=1):
    for e, l in dset.take(num):
        plt.figure(figsize=(10, 10))
        for i in range(len(e)):
            ax = plt.subplot(math.ceil(math.sqrt(len(e))), math.ceil(math.sqrt(len(e))),i+1)
            plt.imshow(e[i].numpy()/255)
            plt.title(l[i].numpy().nonzero()[0].astype(str))
            plt.axis("off")
        plt.show()

def printHistory(history):
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

def train(m: tf.keras.Sequential = model, epochs: int = 10):
    history = m.fit(
        x=d.train_ds,
        epochs=epochs,
        verbose=1,
        validation_data=d.val_ds,
        callbacks= [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.002, patience=3, mode="auto", restore_best_weights=True),
            tf.keras.callbacks.TensorBoard(log_dir="../models/" + s.dir + "/logs", histogram_freq=1),
            c.cm_callback
        ]
    )
    return history

# tfjs.converters.save_keras_model(model, "../models/test1")
# model.save("../models/test1/keras/model.h5", include_optimizer=False)



