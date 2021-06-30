import tensorflow as tf
import tensorflowjs as tfjs
import numpy as np
import matplotlib.pyplot as plt
import math
import model as M

model: tf.keras.Sequential = M.model

def augment(t, label):
    r = tf.random.uniform((), 0, 4, tf.int32)
    t = tf.image.random_flip_left_right(t)
    t = tf.image.rot90(t, r)
    return (t, label)

def plot():
    for e, l in train_ds.take(1):
        plt.figure(figsize=(10, 10))
        for i in range(len(e)):
            ax = plt.subplot(math.ceil(math.sqrt(len(e))),math.ceil(math.sqrt(len(e))),i+1)
            plt.imshow(e[i].numpy()/255)
            plt.title(l[i].numpy().nonzero()[0].astype(str))
            plt.axis("off")
        plt.show()

train_ds: tf.data.Dataset = tf.keras.preprocessing.image_dataset_from_directory(
    directory='C:/sorted/_stash',
    labels="inferred",
    label_mode="categorical",
    image_size=(400,400),
    smart_resize=True,
    batch_size=32,
    shuffle=True,
    seed=529056,
    validation_split=0.15,
    subset="training"
).prefetch(tf.data.AUTOTUNE)

val_ds: tf.data.Dataset = tf.keras.preprocessing.image_dataset_from_directory(
    directory='C:/sorted/_stash',
    labels="inferred",
    label_mode="categorical",
    image_size=(400,400),
    smart_resize=True,
    batch_size=32,
    shuffle=True,
    seed=529056,
    validation_split=0.15,
    subset="validation"
)

# plot()

history = model.fit(
    x=train_ds,
    epochs=10,
    verbose=1,
    # validation_data=val_ds,
    # callbacks= [
    #     tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=3, mode="auto", restore_best_weights=True)
    # ]
)

# summarize history for accuracy
print(history.history)
plt.plot(history.history['categorical_accuracy'])
plt.plot(history.history['val_accuracy'])
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

tfjs.converters.save_keras_model(model, "../models/ot9")

