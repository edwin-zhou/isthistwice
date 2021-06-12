import { environment } from 'src/environments/environment';
import { Injectable } from '@angular/core';
import * as tf from '@tensorflow/tfjs'
import { Tensor3D } from '@tensorflow/tfjs';
import { throwError } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class TfserviceService {
  model: tf.LayersModel
  
  constructor() { 
    this.model = this.loadModel()
  }

  loadModel(): any {
    let mod: tf.LayersModel
    tf.loadLayersModel(environment.modelURL)
    .then((m: tf.LayersModel) => {
      m.summary()
      mod = m
      return mod
    })
    .catch(err => {
      console.log(err)  
    })
    .finally(() => {
      return mod
    })
  }

  bufftoarr(tensor: Tensor3D) {
    return tf.browser.toPixels(tensor)
  }

  /** makes prediction over 1 image */
  predict(tensor: tf.Tensor3D): string {
    let t: tf.Tensor3D[] = [tensor]
    let pred = this.model.predict(tf.stack(t), {batchSize: 1})
    return pred.toString()
  }

  loadImage(img: ImageData): tf.Tensor3D {
    return tf.browser.fromPixels(img)
  }
}
