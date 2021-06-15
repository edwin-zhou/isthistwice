const tf = require('@tensorflow/tfjs-node-gpu')
const blazeface = require('@tensorflow-models/blazeface')
const fs = require('fs')
const path = require('path')

async function main() {
    const model = await blazeface.load()
    console.log(model)
}

/**input tensor and return array */
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

main()
