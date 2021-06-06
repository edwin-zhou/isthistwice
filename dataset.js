const tf = require('@tensorflow/tfjs-node');
const path = require('path');
const fs = require('fs');

var model = require('./model').model
var SAMPLE_SIZE = require('./model').SAMPLE_SIZE
const e = require('express');
var xs, ys, ds

const species = ['Dog', 'Cat']
const IMG_SIZE = require('./model').IMG_SIZE

var numCalled = 0

var indexes = []
species.forEach((name) => {
    let pa = path.join(__dirname, 'petimages', name)
    let filenames = fs.readdirSync(pa)
    indexes.push(filenames.slice(0, 200))
})
indexes.forEach(e => {
    console.log(e.length)
})
var dog = getBatches(indexes[0], species.indexOf('Dog'))
var cat = getBatches(indexes[1], species.indexOf('Cat'))

xs = tf.data.generator(data)
ys = tf.data.generator(labels)
ds = tf.data.zip({xs, ys}).shuffle(SAMPLE_SIZE).batch(SAMPLE_SIZE)

function* data() {
    let index = true
    let dogTensor = dog.next()
    let catTensor = cat.next()
    while (!dogTensor.done && !catTensor.done) {
        numCalled++
        console.log(numCalled)
        let d = dogTensor.value
        let c = catTensor.value
        if (index) {
            yield tf.tensor3d(d)
            dogTensor = dog.next()
            index = false
        } else {
            yield tf.tensor3d(c)
            catTensor = cat.next()
            index = true
        }
    }
    return
}

function* getBatches(arr, speciesIndex) {
    let batch = []
    let ta
    let bool = true
    for(let x=0; x<arr.length; x++) {
        // if (batch.length === SAMPLE_SIZE/2) {
        //     console.log('alo')
        //     let y = batch
        //     batch = []
        //     yield y
        // } else {
            try {
                let filePath = path.join(__dirname, 'petimages', species[speciesIndex], arr[x])
                if (path.extname(filePath) === '.jpeg' || path.extname(filePath) === '.jpg') {
                    tf.tidy(() => {
                        let buff = fs.readFileSync(filePath)
                        let t = tf.node.decodeImage(buff).resizeBilinear(IMG_SIZE)
                        if (t.shape.toString() === IMG_SIZE.concat([3]).toString()) {
                            bool = true
                            ta = t.arraySync()
                        } else {
                            bool = false
                        }
                    })
                    if (bool) {
                        yield ta
                    }
                    ta = null
                }
            } catch (error) {
                // console.log(error)
            }
        // } 
    }
    return
}

function* labels() {
    let index = true
    while(true) {
        // let arr = new Array(SAMPLE_SIZE)
        // arr.fill(0, 0, (SAMPLE_SIZE/2))
        // arr.fill(1, SAMPLE_SIZE/2)
        // yield tf.oneHot(arr, 2)
        yield index? tf.oneHot(0, 2) : tf.oneHot(1, 2)
        index = !index
    }
}

// ds.take(1).forEachAsync(e => {
//     let y = e.ys.arraySync()
//     let a = e.xs.arraySync()

//     for (let x=0;x<a.length;x++) {
//         tf.node.encodeJpeg(tf.tensor3d(a[x]), 'rgb')
//         .then(arr => {
//             fs.writeFileSync(path.join(__dirname, 'models', x.toString() + '.jpg'), arr)
//             console.log(y[x])
//         })
//         .catch(err => {
//             console.log('niktram')
//         })
//     }
// })


model.fitDataset(ds, {
    epochs: 1,
})
.then(history => {
    model.save('file://./models/model1')
    .then(res => {
        console.log(res)
        console.log(history.history)
    })
    .catch(err => {
            console.log(err)
     })
})
.catch(err => {
    console.log(err)
})

module.exports = {ds, SAMPLE_SIZE}

