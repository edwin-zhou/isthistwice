const tf = require('@tensorflow/tfjs-node');
const path = require('path');
const fs = require('fs');
const express = require('express');
const rateLimit = require("express-rate-limit");
const app = express()
const PORT = process.env.PORT || 3000

var config = require('./settings')
var model
// var dataset = require('./dataset.js');

app.use((req,res,next) => {
    res.header('Access-Control-Allow-Origin', '*');
    res.header('Access-Control-Allow-Credentials', 'true');
    res.header('Access-Control-Allow-Methods', 'GET,HEAD,PUT,PATCH,POST,DELETE');
    res.header('Access-Control-Expose-Headers', 'Content-Length');
    res.header('Access-Control-Allow-Headers', 'Accept, Authorization, Content-Type, X-Requested-With, Range');
    console.log(`${req.method} for ${req.url}`)
    next()
})

app.use(rateLimit({
	windowMs: 1000, // 1 second
	max: 2, // limit each IP to 2 requests per windowMs
}))

app.use(express.json({limit: '50mb'}))
app.use(express.urlencoded({extended: true}))

// app.use('/models', express.static(__dirname + '/models'))
app.post('/models/:modelname', (req, res, next) => {
    if (req.params.modelname == config.MODEL_NAME  && req.body.image) {
        let p = tf.tidy(() => {return model.predict(tf.tensor4d(req.body.image, [req.body.image.length, 256, 256, 3])).arraySync()}) 
        res.status(200).send({pred: p})
        tf.dispose(p)
    }
})

app.use('/settings', (req, res, next) => {
    res.json(config)
})

// serve angular
app.use(express.static(path.join(__dirname, 'frontend', 'dist', 'frontend')))

app.get('/*', async (req, res) => {
    res.sendFile(path.resolve(__dirname, 'frontend', 'dist', 'frontend', 'index.html'));
});

tf.loadLayersModel(`file://./models/${config.MODEL_NAME}/model.json`)
.then(mod => {
    model = mod
    app.listen(PORT, () => {
        console.log(`running on port ${PORT}`)
    })
})
.catch(err => {
    console.log(err)
})


