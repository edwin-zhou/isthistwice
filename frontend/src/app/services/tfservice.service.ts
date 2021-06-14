import { environment } from 'src/environments/environment';
import { Injectable, OnInit } from '@angular/core';
import * as tf from '@tensorflow/tfjs'
import { Tensor3D } from '@tensorflow/tfjs';

@Injectable({
  providedIn: 'root'
})
export class TfserviceService {
  model!: tf.LayersModel
  SPECIES!: string[]
  IMG_SIZE! : [number, number]
  
  constructor() { 
    this.loadModel()

    fetch(environment.mainURL + '/settings')
    .then(res => {
      res.json()
      .then(obj => {
        this.SPECIES = obj.SPECIES
        this.IMG_SIZE = obj.IMG_SIZE
      })
      .catch(err => {
        console.log(err)
      })
    })
    .catch(err => {

    })
  }

  async loadModel() {
    this.model = await tf.loadLayersModel(environment.modelURL)
  }

  bufftoarr(tensor: Tensor3D) {
    return tf.browser.toPixels(tensor)
  }

  /** makes prediction over 1 image */
  predict(tensor: tf.Tensor3D): tf.Tensor {
    let t: number[][][][] = [this.normalize(tensor)]
    let pred: tf.Tensor | tf.Tensor[] = this.model.predict(tf.tensor4d(t), {batchSize: 1})
    return pred as tf.Tensor
  }

  loadImage(img: any): tf.Tensor3D {
    return tf.browser.fromPixels(img).resizeBilinear(this.IMG_SIZE)
  }

  normalize(tensor: tf.Tensor3D): number[][][] {
    let t = tf.tidy(() => {
      if (tensor.shape.toString() === this.IMG_SIZE.concat([3]).toString()) {
        let ta = tensor.arraySync()

        let m = tf.moments(tensor)

        let mean = m.mean.dataSync()[0]
        let stdev = Math.max(tf.sqrt(m.variance).dataSync()[0], 1/Math.sqrt(this.IMG_SIZE[0]*this.IMG_SIZE[1]*3)) 

        for (let x=0;x<this.IMG_SIZE[0];x++) {
          for (let y=0;y<this.IMG_SIZE[1];y++) {
              for (let z=0;z<3;z++) {
                  let v = ta[x][y][z]
                  ta[x][y][z] = (v - mean) / stdev
              }
          }
        }
        return ta
      } else {
        return []
      } 
    })
    return t
  }
}
