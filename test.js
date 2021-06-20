const tf = require('@tensorflow/tfjs-node-gpu')
const fs = require('fs')
const path = require('path')

let pa = path.join(__dirname, 'images', 'chae', 'IMG_2956.JPG')

let t = tf.node.decodeJpeg(fs.readFileSync(pa))
t = tf.cast(t, 'float32')
t = tf.image.flipLeftRight(t.expandDims())


tf.node.encodeJpeg(t, 'rgb')
.then(e => {
    fs.writeFileSync(path.join(__dirname, 'images', 'resized', '0' + '.jpg'), e)
})



console.log(t.shape)