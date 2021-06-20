const tf = require('@tensorflow/tfjs-node-gpu')
var blazeface = require('@tensorflow-models/blazeface')
const fs = require('fs')
const path = require('path')
const config = require('./settings')

async function main() {
    blazeface.load({
        maxFaces: 1,
        inputWidth: config.IMG_SIZE[0],
        inputHeight: config.IMG_SIZE[1],
    })
    .then(model => {
        // console.log(model)
        let files = fs.readdirSync(path.join(__dirname, 'images', 'chae'))
        let s = [0,0,0]

        files.forEach(filename => {

            let t = loadImage('chae', filename)

            if (t.shape[0] > s[0]) {
                console.log(filename)
                console.log(t.shape)
                s = t.shape
            }
            // model.estimateFaces(t, false)
            // .then(val => {
            //     let tt = tf.image.cropAndResize(tf.expandDims(t), tf.tensor2d([val[0].topLeft.map(e => {return e/config.IMG_SIZE[0]}).concat(val[0].bottomRight.map(e => {return e/config.IMG_SIZE[0]}))]), [0], config.IMG_SIZE).unstack()
            //     tt.forEach(e => {
            //         e.print
            //     })
            //     // tf.node.encodeJpeg(tt[0], 'rgb')
            //     // .then(a => {
            //     //     fs.writeFileSync(path.join(__dirname, 'images', 'resized', filename + '.jpg'), a)
            //     // })
            //     // .catch(err => {
            //     //     console.log(err)
            //     // })
            // })
            // .catch(err => {
            //     console.log(err)
            // })
        })
        console.log(s)
        
    })
    .catch(err => {
        console.log(err)
    })
}

/**takes images subfolder and array of filenames, returns tensor4d */
function loadImage(dir, filename) {
    let pa = path.join(__dirname, 'images', dir,  filename)

    try {
        let buff = fs.readFileSync(pa)
        let t = tf.node.decodeImage(buff)
        let tt = t.pad(getPadding(t.shape))

        // tf.node.encodeJpeg(tt, 'rgb')
        // .then(a => {
        //     fs.writeFileSync(path.join(__dirname, 'images', 'resized', filename + '.jpg'), a)
        // })
        // .catch(err => {
        //     console.log(err)
        // })

        return tt
    } catch (error) {
        console.log(error)        
    }
}

function getPadding(shape) {
    let arr = [[0,0],[0,0],[0,0]]
    let dif = shape[0]-shape[1]

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
