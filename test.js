const tf = require('@tensorflow/tfjs-node')
const fs = require('fs')
const path = require('path')

let pa = path.join(__dirname, 'images/test')

c = fs.readdirSync(pa)

// let t = tf.node.decodeJpeg(fs.readFileSync(pa), 3)

// tf.node.encodeJpeg(t, 'rgb')
// .then(e => {
//     fs.writeFileSync(path.join(__dirname, 'images', 'test', '0' + '.jpg'), e)
// })



console.log(c)