const tf = require('@tensorflow/tfjs-node')
var blazeface = require('@tensorflow-models/blazeface')
const fs = require('fs')
const path = require('path')
const mergedirs = require('merge-dirs')
const config = require('./settings')

var readir = 'D:/node-app/isthischae/images/processed/Tzuyu'
var writedir = 'images/temp'
var sortingdir = 'D:/node-app/isthischae/images/sorting'

var model

async function main() {
    model = await tf.loadLayersModel('file://./models/' + config.MODEL_NAME + '/model.json')
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
        let f = fs.readdirSync(readir)
        for (let x=0;x<f.length;x++) {
            // console.log(`${x} / ${f.length}  ${f[x]}`)
            await doomz(path.join(readir,f[x]), blaze, false, true)
        }

        // for (let x=0;x<f.length;x++) {
        //     filenames = fs.readdirSync(path.join(readir, f[x]))

        //     for (let y=5000;y<8000;y++) {
        //         let t = loadImage(readir +"/"+ f[x], filenames[y])

        //         if (t) {
        //             await saveFace(blaze, t, f[x], filenames[y])
        //         }
        //     }

        // }
}

var count = 0

// read directories recustively and deletes empty dirs -- if saveface == true saves cropped face
async function doomz(pa, blaze, saveface, sortface) {
    if (fs.lstatSync(pa).isDirectory()) {
        let name = path.basename(pa)
        // if (config.LABELS.includes(path.basename(pa))) {
        //     console.log(name)
        //     mergedirs.default(pa, path.join(readir, name), 'skip');
        // }
        let d = fs.readdirSync(pa)
        if (d.length) {
            for (let x=0;x<d.length;x++) {
                await doomz(path.join(pa, d[x]), blaze, saveface, sortface)
            }
        } else {
            fs.rmdirSync(pa, {recursive:true, maxRetries: 5})
            console.log(`dir ${pa}`)
        }
   
    } 
    else if (fs.lstatSync(pa).isFile()) {
        count++
        // fs.renameSync(pa, pa.split(".")[0] + count.toString() + "pwinss1" + path.extname(pa))
        let t
        try {
            let buff = fs.readFileSync(pa)
            t = tf.node.decodeImage(buff, 3)

            if (t.shape.length === 3 && Math.min(t.shape[0], t.shape[1]) >= config.IMG_SIZE[0]) {
                // let b = t.unstack()
                // b.forEach((e,i) => {
                //     saveFace(model, e.pad(getPadding(e.shape)), "Tzuyu", filename+i.toString())
                // })
                if (saveface) {
                    await saveFace(blaze, t, path.basename(path.dirname(pa)), path.join(pa))
                }
                if (sortface) {
                    await sortFace(t, pa)
                }
                // savePred(blaze, model, t, filename)
            } else {
                tf.dispose(t)
            }
        } catch (error) {
            // fs.unlinkSync(pa)
            tf.dispose(t)
            tf.disposeVariables()
            console.log(`err ${pa}`)
            console.log(error)
        }
    }
    return
}

/**takes images subfolder and array of filenames, returns tensor4d */
function loadImage(dir, filename) {
    let pa = path.join(dir, filename)

    try {
        let buff = fs.readFileSync(pa)

        let t = tf.node.decodeImage(buff, 3)
        
        if (t.shape.length === 3) {
            return t
        } else {
            return null
        }

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

async function saveFace(blaze, t, label, filepath) {
    let s = tf.tidy(() => { return t.clone().pad(getPadding(t.shape)).resizeBilinear([256,256]) })
    blaze.estimateFaces(s, false)
    .then(val => {
        if (val.length != 1) {
            fs.unlinkSync(filepath)
            console.log(`file ${filepath}`)
            tf.dispose(t)
            tf.dispose(s)
        } 
        if (val[0] && val.length === 1) {
            // console.log(val[0])
            let tl = val[0].topLeft.map((e, i) => {return e/256 }).reverse()
            let br = val[0].bottomRight.map((e, i) => {return e/256}).reverse()

            let tt = tf.tidy(() => { return tf.image.cropAndResize(tf.expandDims(t.pad(getPadding(t.shape))), tf.tensor2d([tl.concat(br)]), [0], config.IMG_SIZE, 'bilinear').unstack()[0] }) 
            console.log(path.join(__dirname, writedir, label, path.basename(filepath)))
            tf.node.encodeJpeg(tt, 'rgb')
            .then(a => {
                fs.writeFileSync(path.join(__dirname, writedir, label, path.basename(filepath)), a)
            })
            .catch(err => {
                console.log(err)
            })
            .finally(() => {
                tf.dispose(tt)
                tf.dispose(t)
                tf.dispose(s)
            })
        } else {
            tf.dispose(t)
            tf.dispose(s)
        }
    })
    .catch(err => {
        console.log(err)
    })
}

async function sortFace(t, filepath) {
    label = path.basename(path.dirname(filepath))
    pred = model.predict(tf.expandDims(t)).squeeze().arraySync()
    certainty = Math.max(...pred)
    if (certainty <= .4) {
        fs.renameSync(filepath, path.join(sortingdir, path.basename(filepath)))
    }
    else if (pred.indexOf(certainty) != config.LABELS.indexOf(label) && certainty >= .7) {
        fs.renameSync(filepath, path.join(sortingdir, config.LABELS[pred.indexOf(certainty)], path.basename(filepath)))
    } else if (certainty <= .7 && certainty > .4) {
        console.log(path.join(sortingdir, label, path.basename(filepath)))
        fs.renameSync(filepath, path.join(sortingdir, label, path.basename(filepath)))
    }
    tf.dispose(pred)
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
