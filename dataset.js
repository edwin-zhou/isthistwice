const tf = require('@tensorflow/tfjs-node-gpu');
const path = require('path');
const fs = require('fs');

var model = require('./model');
var xs, ys, ds

const species = ['Dog', 'Cat']
const SAMPLE_SIZE = 32
const IMG_FORMAT = [350, 350, 3]
const IMG_SIZE = [350, 350]

var numCalled = 0

var indexes = []
species.forEach((name) => {
    let pa = path.join(__dirname, 'petimages', name)
    let filenames = fs.readdirSync(pa)
    indexes.push(filenames)
})
var dog = getBatches(indexes[0], species.indexOf('Dog'))
var cat = getBatches(indexes[1], species.indexOf('Cat'))
var dogTensor = dog.next()
var catTensor = cat.next()

xs = tf.data.generator(data)
ys = tf.data.generator(labels)
ds = tf.data.zip({xs, ys})

function* data() {
    while (!dogTensor.done && !catTensor.done) {
        numCalled++
        console.log(numCalled)
        let d = dogTensor.value
        let c = catTensor.value
        yield d.concat(c) 
        dogTensor = dog.next()
        catTensor = cat.next()
    }
    return
}

function* getBatches(arr, speciesIndex) {
    let batch = []
    for(x=0; x<arr.length; x++) {
        if (batch.length === SAMPLE_SIZE/2) {
            console.log('alo')
            let y = batch
            batch = []
            yield y
        } else {
            try {
                let filePath = path.join(__dirname, 'petimages', species[speciesIndex], arr[x])
                if (path.extname(filePath) === '.jpeg' || path.extname(filePath) === '.jpg') {
                    tf.tidy(() => {
                        let buff = fs.readFileSync(filePath)
                        let t = tf.node.decodeImage(buff).resizeBilinear(IMG_SIZE).arraySync()
                        batch.push(t)
                    })
                }
            } catch (error) {
                console.log(error)
            }
        } 
    }
    return
}

function* labels() {
    while(true) {
        let arr = new Array(SAMPLE_SIZE)
        arr.fill(0, 0, (SAMPLE_SIZE/2))
        arr.fill(1, SAMPLE_SIZE/2)
        yield tf.oneHot(arr, 2)
    }
}

ds.forEachAsync(e => {
    console.log(numCalled)
})


// model.fitDataset(ds, {
//     epochs: 10,
// })
// .then(history => {
//     return history
// })
// .catch(err => {
//     return err
// })

module.exports = ds

