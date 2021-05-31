const tf = require('@tensorflow/tfjs-node');

const species = ['Cat', 'Dog']

var model = tf.sequential()

model.add(
    tf.layers.conv2d({
        inputShape: [350, 350, 3],
        batchSize: 32,
        kernelSize: 50,
        filters: 8,
        strides: 1,
        activation: 'relu',
        kernelInitializer: 'varianceScaling'
    })
)

model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));
  
// Now we flatten the output from the 2D filters into a 1D vector to prepare
// it for input into our last layer. This is common practice when feeding
// higher dimensional data to a final classification output layer.
model.add(tf.layers.flatten());

// Our last layer is a dense layer which has 10 output units, one for each
// output class (i.e. 0, 1, 2, 3, 4, 5, 6, 7, 8, 9).
model.add(tf.layers.dense({
  units: species.length,
  kernelInitializer: 'varianceScaling',
  activation: 'softmax'
}));

model.compile({
    optimizer: tf.train.adam(0.1),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
});

module.exports = model