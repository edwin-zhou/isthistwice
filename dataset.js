const tf = require('@tensorflow/tfjs-node-gpu');
const path = require('path');
const fs = require('fs');

var bol = true

var trainSet = [1,3,5,6,87,1,5,8,9]
var trainLabels = [2,5,7,4,169,2,12,17,20]

var testSet = [5,6,87,1,5,8,9]
var testLabels = [2,5,7,4,169,2,12]

var trainDataset
var testDataset

for (x=0;x<trainSet.length;x++) {
    trainSet[x] = {xs: x}
    trainLabels[x] = {ys: x}

}


for (y=0;y<testSet.length;y++) {
    testSet[x] = {xs: x}
    testLabels[x] = {ys: x}
}


var model = tf.sequential()

model.add(
    tf.layers.dense({
        inputDim: 1,
        activation: 'relu',
        units: 1
    })
)

model.compile({
    optimizer: tf.train.adam(),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
});

model.fitDataset(tf.data.zip([tf.data.array(trainSet), tf.data.array(trainLabels)]), {
    batchesPerEpoch: 3,
    validationData: tf.data.zip([tf.data.array(testSet, tf.data.array(testLabels))]),
    epochs: 10
})
.then(history => {
    console.log(history)
})
.catch(err => {
    console.log(err)
})

