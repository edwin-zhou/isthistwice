const tf = require('@tensorflow/tfjs-node');
const path = require('path');
const fs = require('fs');
const e = require('express');

const IMG_SIZE = require('./model').IMG_SIZE
var model
var dataset = require('./dataset.js');

tf.loadLayersModel('file://./models/model1/model.json')
.then(mod => {
    model = mod
    let pred = model.predict(loadImage('Cat', '0.jpg'), {batchSize: 1})
    pred.print()    
})
.catch(err => {
    console.log(err)
})

function loadImage(dir, filename) {
    let pa = path.join(__dirname, 'petimages', dir,  filename)

    let buff = fs.readFileSync(pa)

    try {
        let t = tf.node.decodeImage(buff).resizeBilinear(IMG_SIZE)
        t = dataset.normalize(t)
        return tf.tensor4d([t], [1,200,200,3])
    } catch (error) {
        console.log(error)        
    }
}


