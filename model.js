const tf = require('@tensorflow/tfjs-node');
var config = require('./settings')

const species = config.LABELS

var model = tf.sequential()
const BATCH_SIZE = config.BATCH_SIZE
const IMG_SIZE = config.IMG_SIZE

model.add(
    tf.layers.batchNormalization({
        inputShape: IMG_SIZE.concat([3])
    })
)

model.add(
    tf.layers.conv2d({
        kernelSize: 7,
        padding: 'same',
        filters: 16,
        activation: 'relu',
    })
)

model.add(
    tf.layers.conv2d({
        kernelSize: 5,
        padding: 'same',
        filters: 32,
        activation: 'relu',
    })
)

model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));

model.add(
    tf.layers.conv2d({
        kernelSize: 3,
        padding: 'same',
        filters: 64,
        activation: 'relu',
    })
)

model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));

model.add(
    tf.layers.conv2d({
        kernelSize: 2,
        padding: 'same',
        filters: 64,
        activation: 'relu',
    })
)

model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));
  
model.add(tf.layers.flatten());

model.add(tf.layers.dense({
    units: 10,
    activation: 'relu'
}));

model.add(tf.layers.dense({
  units: species.length,
  activation: 'softmax'
}));

model.compile({
    optimizer: tf.train.adam(0.00001),
    loss: 'binaryCrossentropy',
    metrics: ['accuracy'],
});

model.summary()

module.exports = {model, BATCH_SIZE, IMG_SIZE}