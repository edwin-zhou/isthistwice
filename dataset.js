const tf = require('@tensorflow/tfjs-node');
const path = require('path');
const fs = require('fs');
const e = require('express');

var model = require('./model').model
var BATCH_SIZE = require('./model').BATCH_SIZE
const IMG_SIZE = require('./model').IMG_SIZE

const species = ['Dog', 'Cat']
var pointers = []
var xs, ys, ds
var vxs = []
var vys = []

xs = tf.data.generator(data)
ys = tf.data.generator(labels)
ds = tf.data.zip({xs, ys}).batch(BATCH_SIZE)

var files = []
species.forEach((name, i) => {
    let pa = path.join(__dirname, 'petimages', name)
    let filenames = fs.readdirSync(pa).filter(file => {return path.extname(file) === '.jpg'})
    pointers.push(getSample(i))
    files.push(filenames.slice(0, 5))
})

let numCalled = 0
var curIndex = 0
function* data() {
    while (true) {
        for (let x=0;x<files.length;x++) {
            let s = pointers[x].next()
            if (!s.done) {
                curIndex = x
                yield tf.tensor3d(s.value)
            } else {
                files.forEach((e, i) => {
                    shuffle(e)
                    pointers[i] = getSample(i)
                })
                break
            }
        }
    }
}

function* getSample(speciesIndex) {
    for(let x=0; x<files[speciesIndex].length; x++) {
        try {
            let filePath = path.join(__dirname, 'petimages', species[speciesIndex], files[speciesIndex][x])
            if (path.extname(filePath) === '.jpeg' || path.extname(filePath) === '.jpg') {
                let s = tf.tidy(() => {
                    let buff = fs.readFileSync(filePath)
                    let t = tf.node.decodeImage(buff).resizeBilinear(IMG_SIZE)
                    if (t.shape.toString() === IMG_SIZE.concat([3]).toString()) {
                        return normalize(t)
                    } else {
                        return
                    }
                })
                yield s
            }
        } catch (error) {
            console.log(error)
        }
    }
    return
}

function* labels() {
    while(true) {
        yield tf.oneHot(curIndex, species.length) 
    }
}

function normalize(tensor) {
    let t = tf.tidy(() => {
        if (tensor.shape.toString() === IMG_SIZE.concat([3]).toString()) {
            let ta = tensor.arraySync()

            let m = tf.moments(tensor)

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
            return ta
        } 
    })
    return t
}

function shuffle(arr) {
    for (let x=arr.length-1;x>0;x--) {
        let ind = Math.floor(Math.random() * (x+1))
        let a = arr[x]
        arr[x] = arr[ind]
        arr[ind] = a
    }
}

// ds.take(2).forEachAsync(e => {
//     e.xs.arraySync().forEach((arr, i) => {
//         tf.node.encodeJpeg(tf.tensor3d(arr), 'rgb')
//         .then(a => {
//             fs.writeFileSync(path.join(__dirname, 'models', i.toString() + '.jpg'), a)
//         })
//         .catch(err => {
//             console.log('niktram')
//         })
//     })
//     e.ys.print()
// })


// model.summary()

// model.fitDataset(ds, {
//     epochs: 15,
//     batchesPerEpoch: 1,
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

module.exports = {ds, SAMPLE_SIZE: BATCH_SIZE , normalize}

