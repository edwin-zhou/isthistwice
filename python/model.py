import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.BatchNormalization(input_shape=(256,256,3)),
    tf.keras.layers.Conv2D(
        # input_shape=[256,256,3],
        filters=64,
        kernel_size=5,
        strides=1,
        activation=tf.keras.activations.relu,
        # kernel_initializer=tf.keras.initializers.VarianceScaling(scale=0.1, mode='fan_in', distribution='uniform')
    ),
    tf.keras.layers.Conv2D(
        filters=128,
        kernel_size=3,
        strides=1,
        activation=tf.keras.activations.relu,
        # kernel_initializer=tf.keras.initializers.VarianceScaling(scale=0.1, mode='fan_in', distribution='uniform')
    ),
    tf.keras.layers.MaxPool2D(
        pool_size=5
    ),
    # tf.keras.layers.Conv2D(
    #     filters=256,
    #     kernel_size=3,
    #     strides=1,
    #     activation=tf.keras.activations.relu,
    #     # kernel_initializer=tf.keras.initializers.VarianceScaling(scale=0.1, mode='fan_in', distribution='uniform')

    # ),
    # tf.keras.layers.Conv2D(
    #     filters=256,
    #     kernel_size=3,
    #     strides=1,
    #     activation=tf.keras.activations.relu,
    #     # kernel_initializer=tf.keras.initializers.VarianceScaling(scale=0.1, mode='fan_in', distribution='uniform')

    # ),

    #     tf.keras.layers.Conv2D(
    #     filters=265,
    #     kernel_size=3,
    #     strides=1,
    #     activation=tf.keras.activations.relu,
    #     # kernel_initializer=tf.keras.initializers.VarianceScaling(scale=0.1, mode='fan_in', distribution='uniform')

    # ),
    tf.keras.layers.Conv2D(
        filters=256,
        kernel_size=3,
        strides=1,
        activation=tf.keras.activations.relu,
        # kernel_initializer=tf.keras.initializers.VarianceScaling(scale=0.1, mode='fan_in', distribution='uniform')
    ),
        tf.keras.layers.Conv2D(
        filters=8,
        kernel_size=3,
        strides=1,
        activation=tf.keras.activations.relu,
        # kernel_initializer=tf.keras.initializers.VarianceScaling(scale=0.1, mode='fan_in', distribution='uniform')
    ),
    tf.keras.layers.MaxPool2D(
        pool_size=2
    ),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=50 ,activation=tf.keras.activations.relu),
    # tf.keras.layers.Dense(units=1024,activation=tf.keras.activations.relu),
    tf.keras.layers.Dense(
        units=9,
        activation=tf.keras.activations.softmax,
        # kernel_initializer=tf.keras.initializers.VarianceScaling(scale=0.1, mode='fan_in', distribution='uniform')
    )
])
model.compile(
    optimizer=tf.optimizers.Adam(1e-4),
    loss=tf.keras.losses.categorical_crossentropy,
    metrics=[tf.keras.metrics.categorical_accuracy]
)
