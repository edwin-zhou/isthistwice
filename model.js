const tf = require('@tensorflow/tfjs-node');

const species = ['Cat', 'Dog']

var model = tf.sequential()
const SAMPLE_SIZE = 32
const IMG_SIZE = [250, 250]

model.add(
    tf.layers.conv2d({
        inputShape: IMG_SIZE.concat([3]),
        batchSize: SAMPLE_SIZE,
        kernelSize: 25,
        filters: 32,
        activation: 'relu',
    })
)

model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));

model.add(
    tf.layers.conv2d({
        kernelSize: 25,
        filters: 64,
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
    optimizer: tf.train.adam(),
    loss: 'binaryCrossentropy',
    metrics: ['accuracy'],
});

module.exports = {model, SAMPLE_SIZE, IMG_SIZE}