const tf = require('@tensorflow/tfjs-node-gpu')

var ok = []

tf.tidy(() => {
    for (x=0;x<5;x++) {
        ok.push(tf.tensor1d([1,2,3,4]))
    }
})



console.log(tf.stack(ok))