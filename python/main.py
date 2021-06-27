import tensorflow as tf
import numpy as np

ds = tf.keras.preprocessing.image_dataset_from_directory(
    directory='D:/node-app/isthischae/images/processed',
    labels="inferred",
    label_mode="categorical",
    shuffle=True,
).prefetch(32)

model = tf.keras.Sequential()

model.add(
    tf.keras.layers.BatchNormalization(
        input_shape=[256,256,3]
    )
)

model.add(
    tf.keras.layers.Conv2D(
        filters=16,
        kernel_size=32,
        strides=1,
        padding="same",
        activation=tf.keras.activations.relu,
    )
)

model.add(
    tf.keras.layers.MaxPool2D(pool_size=2)
)

model.add(
    tf.keras.layers.Flatten()
)

model.add(
    tf.keras.layers.Dense(
        units=9,
        activation=tf.keras.activations.softmax
    )
)

model.compile(
    optimizer=tf.optimizers.Adam(0.00001),
    loss=tf.keras.losses.categorical_crossentropy,
    metrics=['accuracy']
)

model.summary()

h = model.fit(
    x=ds,
    epochs=20,
    verbose=1,
)

print(h.history)
