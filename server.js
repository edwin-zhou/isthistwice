const tf = require('@tensorflow/tfjs-node');
const path = require('path');
const fs = require('fs');
const isCorrupted = require('is-corrupted-jpeg')
const PNG = require('pngjs2').PNG;
const e = require('express');
const shuffle = require('knuth-shuffle-seeded')

const species = ['Cat', 'Dog']

var model = require('./model.js')
var dataset = require('./dataset.js')

async function train() {
    console.log('begin fit')

    // const metrics = ['loss', 'val_loss', 'acc', 'val_acc'];
    // const container = {
    //   name: 'Model Training', tab: 'Model', styles: { height: '1000px' }
    // };
    // const fitCallbacks = tfvis.show.fitCallbacks(container, metrics);


    model.fitDataset(dataset, {
        epochs: 10,
    })
    .then(history => {
        return history
    })
    .catch(err => {
        return err
    })
}

// train()
// .then(his => {
//     console.log(his)
// })
// .catch(err => {
//     console.log(err)
// })

dataset.take(2).forEachAsync(e => {
    console.log(e)
})