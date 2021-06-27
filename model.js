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
        kernelSize: 32,
        padding: 'same',
        filters: 32,
        strides: 2,
        activation: 'relu',
        kernelInitializer: 'varianceScaling'
    })
)

model.add(tf.layers.maxPooling2d({poolSize: 2}));

model.add(
    tf.layers.conv2d({
        kernelSize: 15,
        filters: 50,
        strides: 1,
        activation: 'relu',
        kernelInitializer: 'varianceScaling'
    })
)

model.add(tf.layers.maxPooling2d({poolSize: 2}));

model.add(
    tf.layers.conv2d({
        kernelSize: 2,
        filters: 4,
        strides: 1,
        activation: 'relu',
        kernelInitializer: 'varianceScaling'
    })
)

model.add(tf.layers.maxPooling2d({poolSize: 2}));

// model.add(
//     tf.layers.conv2d({
//         kernelSize: 2,
//         padding: 'same',
//         filters: 8,
//         strides: 2,
//         activation: 'relu',
//     })
// )

// model.add(tf.layers.maxPooling2d({poolSize: 2}));

model.add(tf.layers.flatten());

model.add(tf.layers.dense({
  units: config.LABELS.length,
  kernelInitializer: 'varianceScaling',
  activation: 'softmax'
}));

model.compile({
    optimizer: tf.train.adam(0.00001),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
});

// model.summary()

module.exports = {model}