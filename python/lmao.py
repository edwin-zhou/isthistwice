import tensorflow as tf
import tensorflowjs as tfjs
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import io
import itertools

import settings as s
from model import model as model
import data as d
import confusion as c

test_pred_raw = model.predict(d.val_ds, steps=None)
test_pred = np.argmax(test_pred_raw, axis=1)

labels_raw = y = np.concatenate([y for x, y in d.val_ds], axis=0)
labels = np.argmax(labels_raw, axis=1)

c: int = 0
for x in range(d.val_ds.cardinality().numpy()):
    print(test_pred[x])
    print(labels[x])
    print(" ")
    if (test_pred[x] == labels[x]):
        c = c+1 
        print("")
print(c)









