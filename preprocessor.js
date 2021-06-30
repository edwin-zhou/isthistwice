const tf = require('@tensorflow/tfjs-node')
var blazeface = require('@tensorflow-models/blazeface')
const fs = require('fs')
const path = require('path')
const config = require('./settings')


var readir = 'images/sorted/_stash'
var writedir = 'images/sorted/_stash'

async function main() {
    let model = await tf.loadLayersModel('file://./models/' + config.MODEL_NAME + '/model.json')
    let blaze = await blazeface.load({
        maxFaces: 3,
        inputWidth: 128,
        inputHeight: 128,
    })
    // console.log(model)
        // let label = 'Nayeon'
        // let files = fs.readdirSync(path.join(__dirname, readir))
        // files.slice(0).forEach(foldername => {
                //     try {
                // let f = fs.readdirSync(path.join(__dirname, readir, foldername))

        //         f.forEach(filename => {
        //             try {
        //                 let t = loadImage(readir+'/'+label, filename)

        //                 if (t.shape.length === 3 && Math.min(t.shape[0], t.shape[1]) >= config.IMG_SIZE[0]) {
        //                     // let b = t.unstack()
        //                     // b.forEach((e,i) => {
        //                     //     saveFace(model, e.pad(getPadding(e.shape)), "Tzuyu", filename+i.toString())
        //                     // })
        //                     saveFace(blaze, t, label, filename)
        //                     // savePred(blaze, model, t, filename)
        //                 } else {
        //                 }
        //                 tf.dispose(t)
        //             } catch (error) {
                        
        //             }
        //         })

        //     } catch (error) {
                
        //     }

        // })
        let f = fs.readdirSync(path.join(__dirname, 'images', 'sorted', '_stash', 'smurf'))
        for (let x=0;x<f.length;x++) {
            // console.log(`${x} / ${f.length}  ${f[x]}`)
            await doomz(path.join(__dirname, 'images', 'sorted', '_stash', 'smurf', f[x]), blaze)
        }
        // f.slice(0,300).forEach(e => {
        // })
}

var count = 0

async function doomz(pa, blaze) {
    if (fs.lstatSync(pa).isDirectory()) {
        let d = fs.readdirSync(pa)
        if (d.length) {
            for (let x=0;x<d.length;x++) {
                await doomz(path.join(pa, d[x]), blaze)
            }
        } else {
            fs.rmdirSync(pa, {recursive:true, maxRetries: 5})
            console.log(pa)
        }
   
    } 
    // else if (fs.lstatSync(pa).isFile()) {
        // count++
        // fs.renameSync(pa, pa.split(".")[0] + count.toString() + "pwinss" + path.extname(pa))
    //     let t
    //     try {
    //         let buff = fs.readFileSync(pa)
        
    //         t = tf.node.decodeImage(buff, 3)

    //         if (t.shape.length === 3 && Math.min(t.shape[0], t.shape[1]) >= config.IMG_SIZE[0]) {
    //             // let b = t.unstack()
    //             // b.forEach((e,i) => {
    //             //     saveFace(model, e.pad(getPadding(e.shape)), "Tzuyu", filename+i.toString())
    //             // })
    //             await saveFace(blaze, t, '', path.join(pa))
    //             // savePred(blaze, model, t, filename)
    //         } else {
    //             tf.dispose(t)
    //         }
    //     } catch (error) {
    //         fs.unlinkSync(pa)
    //         // console.log(`err ${pa}`)
    //         // console.log(error)
    //     } finally {
    //         tf.dispose(t)
    //         tf.disposeVariables()
    //     }
    // }
    return
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

async function saveFace(blaze, t, label, pa) {
    let s = tf.tidy(() => { return t.clone().pad(getPadding(t.shape)).resizeBilinear([256,256]) })
    blaze.estimateFaces(s, false)
    .then(val => {
        if (val.length != 1) {
            fs.unlinkSync(pa)
            // console.log(`del ${pa}`)
        } else if (val.length === 1) {

        } else {
            console.log('wtrf')
        }
        tf.dispose(s)
        tf.dispose(t)
        // if (val[0]) {
        //     // console.log(val[0])
        //     let tl = val[0].topLeft.map((e, i) => {return e/256 }).reverse()
        //     let br = val[0].bottomRight.map((e, i) => {return e/256}).reverse()

        //     let tt = tf.tidy(() => { return tf.image.cropAndResize(tf.expandDims(t.pad(getPadding(t.shape))), tf.tensor2d([tl.concat(br)]), [0], config.IMG_SIZE, 'bilinear').unstack()[0] }) 

        //     tf.node.encodeJpeg(tt, 'rgb')
        //     .then(a => {
        //         fs.writeFileSync(path.join(__dirname, writedir, label, filename), a)
        //     })
        //     .catch(err => {
        //         console.log(err)
        //     })
        //     .finally(() => {
        //         tf.dispose(t)
        //         tf.dispose(s)
        //     })   
        // }
    })
    .catch(err => {
        tf.dispose(s)
        tf.dispose(t)
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

main()
