const tf = require('@tensorflow/tfjs-node')
var blazeface = require('@tensorflow-models/blazeface')
const fs = require('fs')
const path = require('path')
const config = require('./settings')


var readir = 'images/super'
var writedir = 'images/processed'

async function main() {
    let model = await tf.loadLayersModel('file://./models/' + config.MODEL_NAME + '/model.json')
    tf.tidy(() => {
        blazeface.load({
            maxFaces: 1,
            inputWidth: 128,
            inputHeight: 128,
        })
        .then(blaze => {
            // console.log(model)
            config.LABELS.slice(5).forEach((label, index) => {
                // let label = 'Nayeon'
                let files = fs.readdirSync(path.join(__dirname, readir, label))
                files.slice(0).forEach(filename => {
                    // let t = loadImage(readir, filename)
                    let t = loadImage(readir+'/'+label, filename)

        
                    if (t.shape.length === 3 && Math.min(t.shape[0], t.shape[1]) >= config.IMG_SIZE[0]) {
                        // let b = t.unstack()
                        // b.forEach((e,i) => {
                        //     saveFace(model, e.pad(getPadding(e.shape)), "Tzuyu", filename+i.toString())
                        // })
                        saveFace(blaze, t, label, filename)
                        // savePred(blaze, model, t, filename)
                    } else {
                    }
                })
            })
        })
        .catch(err => {
            console.log(err)
        })
    })
}

/**takes images subfolder and array of filenames, returns tensor4d */
function loadImage(dir, filename) {
    let pa = path.join(__dirname, dir, filename)

    try {
        let buff = fs.readFileSync(pa)

        let t = tf.node.decodeImage(buff, 3)
        return t
        // tf.node.encodeJpeg(tt, 'rgb')
        // .then(a => {
        //     fs.writeFileSync(path.join(__dirname, 'images', 'resized', filename + '.jpg'), a)
        // })
        // .catch(err => {
        //     console.log(err)
        // })
    } catch (error) {
        console.log(error)        
    }
}

async function saveFace(blaze, t, label, filename) {
    let s = tf.tidy(() => { return t.clone().pad(getPadding(t.shape)).resizeBilinear([256,256]) })
    blaze.estimateFaces(s, false)
    .then(val => {
        if (val[0]) {
            // console.log(val[0])
            let tl = val[0].topLeft.map((e, i) => {return e/256 }).reverse()
            let br = val[0].bottomRight.map((e, i) => {return e/256}).reverse()

            let tt = tf.tidy(() => { return tf.image.cropAndResize(tf.expandDims(t.pad(getPadding(t.shape))), tf.tensor2d([tl.concat(br)]), [0], config.IMG_SIZE, 'bilinear').unstack()[0] }) 

            tf.node.encodeJpeg(tt, 'rgb')
            .then(a => {
                fs.writeFileSync(path.join(__dirname, writedir, label, filename), a)
            })
            .catch(err => {
                console.log(err)
            })
            .finally(() => {
                tf.dispose(t)
                tf.dispose(s)
            })   
        }
    })
    .catch(err => {
        console.log(err)
    })
}

async function savePred(blaze, model, t, filename) {
    let s = tf.tidy(() => { return t.clone().pad(getPadding(t.shape)).resizeBilinear([256,256]) })

    blaze.estimateFaces(s, false)
    .then(val => {
        if (val[0]) {
            // console.log(val[0])
            let tl = val[0].topLeft.map((e, i) => {return e/256}).reverse()
            let br = val[0].bottomRight.map((e, i) => {return e/256}).reverse()

            try {
                let tt = tf.tidy(() => { return tf.image.cropAndResize(tf.expandDims(t.pad(getPadding(t.shape))), tf.tensor2d([tl.concat(br)]), [0], config.IMG_SIZE).unstack()[0]}) 

                let pred = model.predict(tt.expandDims(), {batchSize: 1}).arraySync()[0]
    
                // write to highest prob
                let name = config.LABELS[pred.indexOf(Math.max(...pred))]
    
                tf.node.encodeJpeg(t, 'rgb')
                .then(a => {
                    fs.writeFileSync(path.join(__dirname, writedir, name, filename), a)
                })
                .catch(err => {
                    console.log(err)
                })
                .finally(() => {
                    tf.dispose(t)
                    tf.dispose(s)
                    tf.dispose(pred)
                })    
            } catch (error) {
            }
        }
    })
    .catch(err => {
        console.log(err)
    })
}

function getPadding(shape) {
    let arr = [[0,0],[0,0],[0,0]]
    let dif = shape[0]-shape[1]
    
    if (dif === 0 ) {
        return arr
    }

    dif>0? arr[1] = [Math.abs(dif/2), Math.abs(dif/2)] : arr[0] = [Math.abs(dif/2), Math.abs(dif/2)] 

    return arr
}

/**input tensor and return array */
function normalize(tensor) {
    let t = tf.tidy(() => {
        if (tensor.shape.toString() === config.IMG_SIZE.concat([3]).toString()) {
            let ta = tensor.arraySync()

            // let m = tf.moments(tensor)

            // let mean = m.mean.dataSync()[0]
            // let stdev = Math.max(tf.sqrt(m.variance).dataSync()[0], 1/Math.sqrt(config.IMG_SIZE[0]*config.IMG_SIZE[1]*3)) 

            for (let x=0;x<config.IMG_SIZE[0];x++) {
                for (let y=0;y<config.IMG_SIZE[1];y++) {
                    for (let z=0;z<3;z++) {
                        let v = ta[x][y][z]
                        // ta[x][y][z] = (v - mean) / stdev
                        ta[x][y][z] = -1 + (v/255*2)
                    }
                }
            }
            return ta
        } 
    })
    return t
}

main()
