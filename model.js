const tf = require('@tensorflow/tfjs-node');

const species = ['Cat', 'Dog']

var model = tf.sequential()
const BATCH_SIZE = 50
const IMG_SIZE = [200, 200]

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
        kernelSize: 25,
        filters: 16,
        activation: 'relu',
    })
)

model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));
  
model.add(tf.layers.flatten());

model.add(tf.layers.dense({
    units: 64,
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