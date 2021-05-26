const tf = require('@tensorflow/tfjs-node');
const path = require('path');
const fs = require('fs');
const {promisify} = require('util') 
const isCorrupted = require('is-corrupted-jpeg')
const PNG = require('pngjs2').PNG;
const e = require('express');
const shuffle = require('knuth-shuffle-seeded')


var model = require('./model.js')
const species = ['Cat', 'Dog']
var datasetArr = []
var labelArr = []

var dataset

async function loadData(string) {
    let pa = path.join(__dirname, 'petimages', string)
    let filenames = fs.readdirSync(pa)
    // console.log(filenames)
    for (x=0;x<80;x++) {
        let name = filenames[x]
        if (!isCorrupted(path.join(pa, name)) && path.extname(name) === '.jpg') {
            let buff = fs.readFileSync(path.join(pa, name))
            await processImages(buff, string)

        } else {
            // console.log(name)
        }
    }
    return
}

async function processImages(buff, string) {
    // console.log(tf.memory().numTensors)
    tf.tidy(() => {
        let final = []
        let exception = false
        let tensor
        try {
            tensor = tf.node.decodeImage(buff).resizeBilinear([350, 350])
        } catch (error) {
            exception = true
        }
        finally {
            if (!exception && tensor.shape[0] == 350 && tensor.shape[1] == 350) {
                tensor = tf.node.decodeImage(buff).resizeBilinear([350, 350])
                tf.node.encodeJpeg(tensor)
                .then((arr) => {
                    datasetArr.push(tf.node.decodeImage(arr))
                    labelArr.push(species.indexOf(string))
                    // final = [arr, species.indexOf(string)]
                    // (species.indexOf(string) === 0)? [1,0]:[0,1]
                })
                .catch((err) => {
                })
            }
            return
        }
    })

}

async function train() {
    // const metrics = ['loss', 'val_loss', 'acc', 'val_acc'];
    // const container = {
    //   name: 'Model Training', tab: 'Model', styles: { height: '1000px' }
    // };
    // const fitCallbacks = tfvis.show.fitCallbacks(container, metrics);
    console.log(datasetArr.length)

    datasetArr = tf.stack(datasetArr, 0)
    // let l = datasetArr.length
    // datasetArr = tf.tensor2d(datasetArr).reshape([l, 250, 250, 3])
    labelArr = tf.oneHot(labelArr, 2)

    console.log('begin fit')

    model.fit(datasetArr, labelArr, {
        batchSize: 50,
        epochs: 20,
        verbose: 1,
        validationSplit: 0.8,
        shuffle: true
    })
    .then((history) => {
        console.log(history.history)
        // model.save(path.join(__dirname, 'models', Date.now().toString()))
        return history
    })
    .catch(err => {
        return err
    })
}

Promise.all([loadData('Dog'), loadData('Cat')])
.then((arr) => {
    shuffle(datasetArr, 'hello')
    shuffle(labelArr, 'hello')

    train()
})
.catch(err => {
})
