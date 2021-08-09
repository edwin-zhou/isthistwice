import tensorflow as tf
import tensorflowjs as tfjs
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import io
import itertools

import settings as s
from model import model as model
import data as d

file_writer_cm = tf.summary.create_file_writer("../models/" + s.dir + '/logs/cm')

def plot_to_image(figure):
  #     Converts the matplotlib plot specified by 'figure' to a PNG image and
  # returns it. The supplied figure is closed and inaccessible after this call.

  # Save the plot to a PNG in memory.
  buf = io.BytesIO()
  plt.savefig(buf, format='png')
  # Closing the figure prevents it from being displayed directly inside
  # the notebook.
  plt.close(figure)
  buf.seek(0)
  # Convert PNG buffer to TF image
  image = tf.image.decode_png(buf.getvalue(), channels=4)
  # Add the batch dimension
  image = tf.expand_dims(image, 0)
  return image

def plot_confusion_matrix(cm, class_names):
  figure = plt.figure(figsize=(20, 20))
  plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
  plt.title("Confusion matrix")
  plt.colorbar()
  tick_marks = np.arange(len(class_names))
  plt.xticks(tick_marks, class_names, rotation=45)
  plt.yticks(tick_marks, class_names)

  # Compute the labels from the normalized confusion matrix.
  labels = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

  # Use white text if squares are dark; otherwise black.
  threshold = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    color = "white" if cm[i, j] > threshold else "black"
    plt.text(j, i, labels[i, j], horizontalalignment="center", color=color)

  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  return figure

def log_confusion_matrix(epoch, logs):
      
  # # Use the model to predict the values from the validation dataset.
  # test_pred_raw = model.predict(d.val_ds, steps=None)
  # test_pred = np.argmax(test_pred_raw, axis=1)

  # # get labels
  # labels_raw = y = np.concatenate([y for x, y in d.val_ds.take(-1)], axis=0)
  # print(len(labels_raw))
  # ds_labels = np.argmax(labels_raw, axis=1)

  test_pred = []
  ds_labels = []

  for images, labels in d.val_ds.as_numpy_iterator():
    test_pred.append(np.argmax(model.predict(images), axis=1))
    ds_labels.append(np.argmax((labels), axis=1))
  test_pred = [pred for l in test_pred for pred in l]
  ds_labels = [label for l in ds_labels for label in l]
        
  # Calculate the confusion matrix.
  cm = confusion_matrix(ds_labels, test_pred)
  # Log the confusion matrix as an image summary.
  figure = plot_confusion_matrix(cm, class_names=s.class_names)
  cm_image = plot_to_image(figure)

  # Log the confusion matrix as an image summary.
  with file_writer_cm.as_default():
    tf.summary.image("Confusion Matrix", cm_image, step=epoch)

# Define the per-epoch callback.
cm_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix)

log_confusion_matrix(0, "alal")