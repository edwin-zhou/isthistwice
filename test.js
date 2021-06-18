const tf = require('@tensorflow/tfjs-node-gpu')
const fs = require('fs')
const path = require('path')

let pa = path.join(__dirname, 'images', 'faces', 'IMG_6556.jpg')

let t = tf.node.decodeJpeg(fs.readFileSync(pa)).pad([[420,420], [0, 0], [0,0]], 0)

tf.node.encodeJpeg(t, 'rgb')
.then(e => {
    fs.writeFileSync(path.join(__dirname, 'images', 'resized', '0' + '.jpg'), e)
})



console.log(t.shape)