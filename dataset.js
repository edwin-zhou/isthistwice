const tf = require('@tensorflow/tfjs-node-gpu');
const path = require('path');
const fs = require('fs');

var model = require('./model');
var xs, ys, ds

const species = ['Dog', 'Cat']
const BATCH_SIZE = 32
const IMG_FORMAT = [350, 350, 3]
const IMG_SIZE = [350, 350]

xs = tf.data.generator(data)
ys = tf.data.generator(labels)
ds = tf.data.zip({xs, ys}).shuffle(BATCH_SIZE).batch(BATCH_SIZE)

function* data() {
    let indexes = []
    species.forEach((name) => {
        let pa = path.join(__dirname, 'petimages', name)
        let filenames = fs.readdirSync(pa)
        indexes.push(filenames)
    })

    let dog = getBatches(indexes[0], species.indexOf('Dog'))
    let cat = getBatches(indexes[1], species.indexOf('Cat'))

    for (x=0;x<indexes[0].length;x++) {
        let dogTensor = dog.next()
        let catTensor = cat.next()

        if (!dogTensor.done && !catTensor.done) {
            yield dogTensor.value.concat(catTensor.value), [BATCH_SIZE].concat(IMG_FORMAT)
        } else {
            return
        }
    }
}


function* getBatches(arr, speciesIndex) {
    let batch = []
    for(x=0; x<arr.length; x++) {
        if (batch.length === BATCH_SIZE/2) {
            let y =  batch
            batch = []
            yield y
        } else {
            try {
                let buff = fs.readFileSync(path.join(__dirname, 'petimages', species[speciesIndex], arr[x]))
                // tensor = tf.node.decodeImage(buff).resizeBilinear([350, 350])
                batch.push(toArr(buff))
            } catch (error) {
                console.log(error)
            }
        } 
    }
    return
}

function* labels() {
    let arr = new Array(BATCH_SIZE)
    arr.fill(0, 0, (BATCH_SIZE/2))
    arr.fill(1, BATCH_SIZE/2)
    yield tf.oneHot(arr, 2)
}

function toArr(buff) {
    let tensor = tf.node.decodeImage(buff).resizeBilinear(IMG_SIZE)
    tensor.array()
    .then(a => {
        tf.dispose(tensor)
        return a
    })
    .catch(err => {

    })
}

let e = data()

console.log(e.next().value.length)

// model.fitDataset(ds, {
//     epochs: 10
// })
// .then(history => {
//     console.log(history)
// })
// .catch(err => {
//     console.log(err)
// })

