import { environment } from 'src/environments/environment';
import { Injectable, OnInit } from '@angular/core';
import * as tf from '@tensorflow/tfjs'
import { Tensor3D } from '@tensorflow/tfjs';

@Injectable({
  providedIn: 'root'
})
export class TfserviceService {
  model!: tf.LayersModel
  LABELS!: string[]
  IMG_SIZE! : [number, number]
  settings: {
    [key: string]: any,
  } = {}
  
  constructor() { 
    this.loadModel()

    fetch(environment.mainURL + '/settings')
    .then(res => {
      res.json()
      .then(obj => {
        this.settings = obj
        this.LABELS = obj.LABELS
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
    this.model = await tf.loadLayersModel(environment.mainURL + '/models' + this.settings.MODEL_NAME)
    console.log('model loaded')
  }

  bufftoarr(tensor: Tensor3D) {
    return tf.browser.toPixels(tensor)
  }

  /** makes prediction over 1 image */
  predict(tensor: tf.Tensor3D): tf.Tensor {
    let pred: tf.Tensor | tf.Tensor[] = this.model.predict(tensor.expandDims(), {batchSize: 1})
    return pred as tf.Tensor
  }

  loadImage(img: any): tf.Tensor3D {
    let t = tf.browser.fromPixels(img)
    return t.pad(this.getPadding(t.shape)).resizeBilinear(this.IMG_SIZE)
  }

  getPadding(shape: any): [number,number][] {
    let arr: [number,number][] = [[0,0],[0,0],[0,0]]
    let dif = shape[0]-shape[1]

    dif>0? arr[1] = [Math.abs(dif/2), Math.abs(dif/2)] : arr[0] = [Math.abs(dif/2), Math.abs(dif/2)] 

    return arr
  }

  // normalize(tensor: tf.Tensor3D): number[][][] {
  //   let t = tf.tidy(() => {
  //     if (tensor.shape.toString() === this.IMG_SIZE.concat([3]).toString()) {
  //       let ta = tensor.arraySync()

  //       let m = tf.moments(tensor)

  //       let mean = m.mean.dataSync()[0]
  //       let stdev = Math.max(tf.sqrt(m.variance).dataSync()[0], 1/Math.sqrt(this.IMG_SIZE[0]*this.IMG_SIZE[1]*3)) 

  //       for (let x=0;x<this.IMG_SIZE[0];x++) {
  //         for (let y=0;y<this.IMG_SIZE[1];y++) {
  //             for (let z=0;z<3;z++) {
  //                 let v = ta[x][y][z]
  //                 ta[x][y][z] = (v - mean) / stdev
  //             }
  //         }
  //       }
  //       return ta
  //     } else {
  //       return []
  //     } 
  //   })
  //   return t
  // }
}
