import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.BatchNormalization(input_shape=[400,400,3]),
    tf.keras.layers.Conv2D(
        input_shape=[400,400,3],
        filters=32,
        kernel_size=32,
        strides=1,
        activation=tf.keras.activations.relu,
        kernel_initializer=tf.keras.initializers.VarianceScaling(scale=1, mode='fan_in', distribution='uniform')
    ),
    tf.keras.layers.MaxPool2D(
        pool_size=2
    ),
    tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=16,
        strides=1,
        activation=tf.keras.activations.relu,
        kernel_initializer=tf.keras.initializers.VarianceScaling(scale=1, mode='fan_in', distribution='uniform')

    ),
    tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=8,
        strides=1,
        activation=tf.keras.activations.relu,
        kernel_initializer=tf.keras.initializers.VarianceScaling(scale=1, mode='fan_in', distribution='uniform')

    ),
    tf.keras.layers.MaxPool2D(
        pool_size=2
    ),
        tf.keras.layers.Conv2D(
        filters=4,
        kernel_size=7,
        strides=1,
        activation=tf.keras.activations.relu,
        kernel_initializer=tf.keras.initializers.VarianceScaling(scale=1, mode='fan_in', distribution='uniform')

    ),
    tf.keras.layers.Conv2D(
        filters=2,
        kernel_size=5,
        strides=1,
        activation=tf.keras.activations.relu,
        kernel_initializer=tf.keras.initializers.VarianceScaling(scale=1, mode='fan_in', distribution='uniform')

    ),
    tf.keras.layers.MaxPool2D(
        pool_size=2
    ),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(
        units=9,
        activation=tf.keras.activations.softmax,
        kernel_initializer=tf.keras.initializers.VarianceScaling(scale=1, mode='fan_in', distribution='uniform')
    )
])

model.compile(
    optimizer=tf.optimizers.Adam(1e-5),
    loss=tf.keras.losses.categorical_crossentropy,
    metrics=[tf.keras.metrics.categorical_accuracy]
)

model.summary()