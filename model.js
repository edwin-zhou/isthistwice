const tf = require('@tensorflow/tfjs-node');
var config = require('./settings')

var model = tf.sequential()

model.add(
    tf.layers.batchNormalization({
        inputShape: config.IMG_SIZE.concat([3])
    })
)

model.add(
    tf.layers.conv2d({
        kernelSize: 7,
        padding: 'same',
        filters: 8,
        strides: 1,
        activation: 'elu',
    })
)

model.add(tf.layers.maxPooling2d({poolSize: 2}));

model.add(
    tf.layers.conv2d({
        kernelSize: 5,
        padding: 'same',
        filters: 16,
        strides: 1,
        activation: 'relu',
    })
)

model.add(tf.layers.maxPooling2d({poolSize: 2}));

model.add(
    tf.layers.conv2d({
        kernelSize: 4,
        padding: 'same',
        filters: 16,
        strides: 2,
        activation: 'relu',
    })
)

model.add(tf.layers.maxPooling2d({poolSize: 2}));

model.add(
    tf.layers.conv2d({
        kernelSize: 2,
        padding: 'same',
        filters: 2,
        strides: 2,
        activation: 'relu',
    })
)

model.add(tf.layers.maxPooling2d({poolSize: 2}));

model.add(tf.layers.flatten());

model.add(tf.layers.dense({
  units: config.LABELS.length,
  activation: 'softmax'
}));

model.compile({
    optimizer: tf.train.adam(0.00001),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
});

// model.summary()

module.exports = {model}