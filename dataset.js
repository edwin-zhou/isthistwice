const tf = require('@tensorflow/tfjs-node');
const blazeface = require('@tensorflow-models/blazeface')
const path = require('path');
const fs = require('fs');
const e = require('express');

var model = require('./model').model
var config = require('./settings')

var pointers = []
var validationSize = config.VALIDATION_SIZE
var vxs = []
var vys = []

var ds = tf.data.generator(data).map(e => {return augment(e)}).batch(config.BATCH_SIZE)
// var validX = tf.data.array(vxs)
// var validY = tf.data.array(vys)
// var validData = tf.data.zip({xs: validX, ys: validY}).batch(config.BATCH_SIZE)

var files = []
var blaze

async function setup() {
    blaze = await blazeface.load({maxFaces:1})
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
}



async function* data() {
    let numCalled = 0
    let shuffles = 0
    while (true) {
        // console.log('shuffles: ' + shuffles)
        // console.log(numCalled)
        // console.log()
        for (let x=0;x<files.length;x++) {
            let s = await pointers[x].next()

            if (!s.done) {
                // numCalled++
                yield {xs: s.value, ys: tf.oneHot(x, config.LABELS.length)}
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

async function* getSample(arr, speciesIndex) {
    for(let x=0; x<arr.length; x++) {
        try {
            let filePath = path.join(__dirname, config.IMG_PATH, config.LABELS[speciesIndex], arr[x])
                let buff = fs.readFileSync(filePath)
                let t = tf.node.decodeImage(buff, 3)
                if (t.shape.length === 3) {
                    // use whole pic
                    // yield tf.tidy(() => {return t.pad(getPadding(t.shape)).resizeBilinear(config.IMG_SIZE)}) 

                    // face only
                    let e = await getFace(blaze, t)
                    if (e) {
                        yield e
                    }
                } else {
                    tf.dispose(t)
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

async function getFace(bl , t) {
    let b = tf.tidy(() => { return t.clone().pad(getPadding(t.shape)).resizeBilinear([256,256]) }) 

    val = await bl.estimateFaces(b, false)
    if (val.length >= 1) {
        let tl = val[0].topLeft.map((e, i) => {return e/config.IMG_SIZE[0]}).reverse()
        let br = val[0].bottomRight.map((e, i) => {return e/config.IMG_SIZE[1]}).reverse()

        let tt = tf.tidy(() => {
          return tf.image.cropAndResize(tf.expandDims(t), tf.tensor2d([tl.concat(br)]), [0], config.IMG_SIZE, 'bilinear').unstack()[0]
        })
        tf.dispose(b)
        tf.dispose(t)
        return tt
    } else {
        tf.dispose(b)
        tf.dispose(t) 
    } 
}

async function trainModel(model) {
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

/** evaluate model with dataset */
async function evaluateModel(name) {
    let mod = await loadModel('file://./models/' + name + '/' + 'model.json')
    mod.compile({
        optimizer: tf.train.adam(0.00001),
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy'],
    })
    mod.evaluateDataset(ds, {
        batches: 25,
    })
    .then(val => {
        val.forEach(t => {
            t.print()
        })
    })
    .catch(err => {
        console.log(err)
    })
}

async function loadModel(path) {
    return tf.loadLayersModel(path)
}

setup()
.then(() => {

    // ds.forEachAsync(e => {
        // let lab = e.ys.unstack()
        // e.xs.unstack().forEach((arr, i) => {
        //     lab[i].print()        
        //     tf.node.encodeJpeg(arr, 'rgb')
        //     .then(a => {
        //         fs.writeFileSync(path.join(__dirname, 'images', 'test', i.toString() + '.jpg'), a)
        //     })
        //     .catch(err => {
        //         console.log('niktram')
        //     })
        // })
    // })
    tf.loadLayersModel('file://./models/' + config.MODEL_NAME + '/' + 'model.json')
    .then(mod => {
        mod.compile({
            optimizer: tf.train.adam(0.00001),
            loss: 'categoricalCrossentropy',
            metrics: ['accuracy'],
        })
        trainModel(mod)
    })
    // evaluateModel('ot9-v2')

})
.catch(err => {
    console.log(err)
})

module.exports = {ds}

