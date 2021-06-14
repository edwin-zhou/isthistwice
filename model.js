const tf = require('@tensorflow/tfjs-node-gpu');
var config = require('./settings')

const species = config.SPECIES

var model = tf.sequential()
const BATCH_SIZE = config.BATCH_SIZE
const IMG_SIZE = config.IMG_SIZE

model.add(
    tf.layers.conv2d({
        inputShape: IMG_SIZE.concat([3]),
        kernelSize: 25,
        filters: 16,
        activation: 'relu',
    })
)

model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));

model.add(
    tf.layers.conv2d({
        kernelSize: 7,
        filters: 32,
        activation: 'relu',
    })
)

model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));
  
model.add(tf.layers.flatten());

model.add(tf.layers.dense({
    units: 128,
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

module.exports = {model, BATCH_SIZE, IMG_SIZE}