const tf = require('@tensorflow/tfjs-node')

var ok = [1,2,3,4,5,6,7,8,9,10,11,12]
const IMG_SIZE = require('./model').IMG_SIZE

let t = tf.tensor3d(ok, [2,2,3])

let m = tf.moments(t)

// m.mean.print()
// tf.sqrt(m.variance).print()

let stdev = Math.max(tf.sqrt(m.variance).dataSync()[0], 1/Math.sqrt(IMG_SIZE[0]*IMG_SIZE[1]*3)) 

console.log(stdev)