const tf = require('@tensorflow/tfjs-node');
const path = require('path');
const fs = require('fs');
const e = require('express');

var model = require('./model').model
var config = require('./settings')

var pointers = []
var xs, ys, ds
var validationSize = config.VALIDATION_SIZE
var vxs = []
var vys = []

xs = tf.data.generator(data)
ys = tf.data.generator(labels)
ds = tf.data.zip({xs, ys}).map(e => {return augment(e)}).batch(config.BATCH_SIZE)
// ds = tf.data.zip({xs, ys}).shuffle(config.BATCH_SIZE).batch(config.BATCH_SIZE)


var files = []
config.LABELS.forEach((name, i) => {
    let pa = path.join(__dirname, config.IMG_PATH, name)
    let filenames = fs.readdirSync(pa).filter(file => {return path.extname(file) === '.jpg' || path.extname(file) === '.JPG'})
    shuffle(filenames)
    files.push(filenames)
    pointers.push(getSample(files[i],i))
    let count = 0
    // let s = getSample(filenames, i)
    // for (let x=0;count<validationSize/2;x++) {
    //     let e = s.next()
    //     vxs.push(tf.tensor3d(e.value))
    //     vys.push(tf.oneHot(i, species.length))
    //     count++
    // }
    
})


// var validX = tf.data.array(vxs)
// var validY = tf.data.array(vys)
// var validData = tf.data.zip({xs: validX, ys: validY}).batch(config.BATCH_SIZE)

var curIndex = 0
function* data() {
    let numCalled = 0
    let shuffles = 0
    while (true) {
        // console.log('shuffles: ' + shuffles)
        // console.log(numCalled)
        // console.log()

        for (let x=0;x<files.length;x++) {
            let s = pointers[x].next()
            if (!s.done) {
                curIndex = x
                // numCalled++
                yield s.value
            } else {
                files.forEach((e, i) => {
                    shuffle(e)
                    pointers[i] = getSample(e, i)
                })
                shuffles++
                break
            }
        }
    }
}

function* getSample(arr, speciesIndex) {
    for(let x=0; x<arr.length; x++) {
        try {
            let filePath = path.join(__dirname, config.IMG_PATH, config.LABELS[speciesIndex], arr[x])
            let s = tf.tidy(() => {
                let buff = fs.readFileSync(filePath)
                let t = tf.node.decodeImage(buff)
                t = tf.pad(t, getPadding(t.shape)).resizeBilinear(config.IMG_SIZE)
                if (t.shape.toString() === config.IMG_SIZE.concat([3]).toString()) {
                    return t
                } else {
                    return
                }
            })
            if (s) {
                yield s
            }
        } catch (error) {
            // console.log(error)
        }
    }
    return
}

function* labels() {
    while(true) {
        yield tf.oneHot(curIndex, config.LABELS.length) 
    }
}

function getPadding(shape) {
    let arr = [[0,0],[0,0],[0,0]]
    let dif = shape[0]-shape[1]

    if (dif === 0) {
        return arr
    }

    dif>0? arr[1] = [Math.abs(dif/2), Math.abs(dif/2)] : arr[0] = [Math.abs(dif/2), Math.abs(dif/2)] 

    return arr
}

function augment(sample) {
    let t = sample.xs

    // flip
    if (Math.round(Math.random()) === 1) {
        t = t.reverse(1)        
    }

    // rotate
    // if (Math.round(Math.random()) === 1) {
    //     let rad = Math.random()*(2*Math.PI)
    //     let center = Math.random() * .5

    //     t = tf.image.rotateWithOffset(t, rad, 0, center)
    // }
    return {xs: t, ys: sample.ys} 
}

// function normalize(tensor) {
//     let t = tf.tidy(() => {
//         if (tensor.shape.toString() === IMG_SIZE.concat([3]).toString()) {
//             let ta = tensor.arraySync()

//             let m = tf.moments(tensor)

//             let mean = m.mean.dataSync()[0]
//             let stdev = Math.max(tf.sqrt(m.variance).dataSync()[0], 1/Math.sqrt(IMG_SIZE[0]*IMG_SIZE[1]*3)) 

//             for (let x=0;x<IMG_SIZE[0];x++) {
//                 for (let y=0;y<IMG_SIZE[1];y++) {
//                     for (let z=0;z<3;z++) {
//                         let v = ta[x][y][z]
//                         ta[x][y][z] = (v - mean) / stdev
//                     }
//                 }
//             }
//             return ta
//         } 
//     })
//     return t
// }

function shuffle(arr) {
    for (let x=arr.length-1;x>0;x--) {
        let ind = Math.floor(Math.random() * (x+1))
        let a = arr[x]
        arr[x] = arr[ind]
        arr[ind] = a
    }
}


// ds.take(1).forEachAsync(e => {
//     // tf.node.encodeJpeg(tf.tensor3d(arr), 'rgb')
//     // .then(a => {
//     //     fs.writeFileSync(path.join(__dirname, 'images', 'test', i.toString() + '.jpg'), a)
//     // })
//     // .catch(err => {
//     //     console.log('niktram')
//     // })
//     // e.ys.print()
// })


async function trainModel() {
    model.fitDataset(ds, {
        epochs: config.EPOCHS,
        batchesPerEpoch: config.BATCHES_PER_EPOCH,
        // validationData: validData,
        // validationBatches: 2,
        callbacks: {
            onEpochEnd: () => {
                model.save('file://./models/' + config.MODEL_NAME)
                .then(res => {
                    console.log('model saved')
                })
                .catch(err => {
                    console.log(err)
                })
            }
        }
    })
    .then(history => {
        model.save('file://./models/' + config.MODEL_NAME)
        .then(res => {
            console.log(history.history)
            return history
        })
        .catch(err => {
            console.log(err)
        })
    })
    .catch(err => {
        console.log(err)
    })
}

trainModel()

module.exports = {ds}

