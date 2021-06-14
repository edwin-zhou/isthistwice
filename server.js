const tf = require('@tensorflow/tfjs-node-gpu');
const path = require('path');
const fs = require('fs');
const express = require('express');
const app = express()
const PORT = process.env.PORT || 3000

var config = require('./settings')
var model
var dataset = require('./dataset.js');

app.use((req,res,next) => {
    res.header('Access-Control-Allow-Origin', req.get('Origin') || '*');
    res.header('Access-Control-Allow-Credentials', 'true');
    res.header('Access-Control-Allow-Methods', 'GET,HEAD,PUT,PATCH,POST,DELETE');
    res.header('Access-Control-Expose-Headers', 'Content-Length');
    res.header('Access-Control-Allow-Headers', 'Accept, Authorization, Content-Type, X-Requested-With, Range');
    console.log(`${req.method} for ${req.url}`)
    next()
})

app.use('/models', express.static(__dirname + '/models'))

app.use('/settings', (req, res, next) => {
    res.json(config)
})

// tf.loadLayersModel('file://./models/model1/model.json')
// .then(mod => {
//     model = mod
//     let pred = model.predict(loadImage('Cat', '501.jpg'), {batchSize: 1})
//     pred.print()    
// })
// .catch(err => {
//     console.log(err)
// })

function loadImage(dir, filename) {
    let pa = path.join(__dirname, 'petimages', dir,  filename)

    let buff = fs.readFileSync(pa)

    try {
        let t = tf.node.decodeImage(buff).resizeBilinear(config.IMG_SIZE)
        t = dataset.normalize(t)
        return tf.tensor4d([t], [1,200,200,3])
    } catch (error) {
        console.log(error)        
    }
}

app.listen(PORT, () => {
    console.log(`running on port ${PORT}`)
})
