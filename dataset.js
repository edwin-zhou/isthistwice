const tf = require('@tensorflow/tfjs-node');
const path = require('path');
const fs = require('fs');

var model = require('./model').model
var SAMPLE_SIZE = require('./model').SAMPLE_SIZE
const e = require('express');
var xs, ys, ds
var vxs = []
var vys = []
const species = ['Dog', 'Cat']
const IMG_SIZE = require('./model').IMG_SIZE

var numCalled = 0

xs = tf.data.generator(data)
ys = tf.data.generator(labels)
ds = tf.data.zip({xs, ys}).shuffle(SAMPLE_SIZE).batch(SAMPLE_SIZE)

var indexes = []
species.forEach((name) => {
    let pa = path.join(__dirname, 'petimages', name)
    let filenames = fs.readdirSync(pa)
    indexes.push(filenames.slice(0, filenames.length-1-SAMPLE_SIZE))

    // for (let x=0;vxs.length < SAMPLE_SIZE ;x++) {
    //     try {
    //         let filePath = path.join(__dirname, 'petimages', name, filenames[filenames.length-1-x])
    //             let buff = fs.readFileSync(filePath)
    //             let t = tf.node.decodeImage(buff).resizeBilinear(IMG_SIZE)
    //             if (t.shape.toString() === IMG_SIZE.concat([3]).toString()) {
    //                 vxs.push(t)
    //                 vys.push(tf.oneHot(species.indexOf(name), 2))
    //             }
    //     } catch (error) {
    //         console.log(error)
    //     }
    // }
})

var dog = getBatches(indexes[0], species.indexOf('Dog'))
var cat = getBatches(indexes[1], species.indexOf('Cat'))


function* data() {
    let index = true
    let dogTensor = dog.next()
    let catTensor = cat.next()
    while (!dogTensor.done && !catTensor.done) {
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
                            ta = t.arraySync()

                            let m = tf.moments(t)

                            let mean = m.mean.dataSync()[0]
                            let stdev = Math.max(tf.sqrt(m.variance).dataSync()[0], 1/Math.sqrt(IMG_SIZE[0]*IMG_SIZE[1]*3)) 

                            for (let x=0;x<IMG_SIZE[0];x++) {
                                for (let y=0;y<IMG_SIZE[1];y++) {
                                    for (let z=0;z<3;z++) {
                                        let v = ta[x][y][z]
                                        ta[x][y][z] = (v - mean) / stdev
                                    }
                                }
                            }

                            bool = true
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

ds.take(1).forEachAsync(e => {
})

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


// model.fitDataset(ds, {
//     epochs: 3,
//     batchesPerEpoch: 3,
//     validationData: [vxs, vys]
// })
// .then(history => {
//     model.save('file://./models/model1')
//     .then(res => {
//         console.log(res)
//         console.log(history.history)
//     })
//     .catch(err => {
//             console.log(err)
//      })
// })
// .catch(err => {
//     console.log(err)
// })

module.exports = {ds, SAMPLE_SIZE}

