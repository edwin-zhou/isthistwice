import { environment } from 'src/environments/environment';
import { Injectable, OnInit, Output } from '@angular/core';
import * as tf from '@tensorflow/tfjs'
import * as blazeface from '@tensorflow-models/blazeface'
import { Tensor3D } from '@tensorflow/tfjs';
import { BehaviorSubject } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class TfserviceService {
  model!: tf.LayersModel
  blaze!: blazeface.BlazeFaceModel

  settings: {
    [key: string]: any,
  } = {}
  
  private _modelLoaded: BehaviorSubject<boolean> = new BehaviorSubject(false as boolean)
  modelLoaded = this._modelLoaded.asObservable()

  constructor() { 
    fetch(environment.mainURL + '/settings')
    .then(res => {
      res.json()
      .then(obj => {
        this.settings = obj
        console.log(obj)
        this.loadModels()
        .then(() => {
          this._modelLoaded.next(true)
        })
        .catch(err => {
          this._modelLoaded.next(false)
        })
      })
      .catch(err => {
        console.log(err)
      })
    })
    .catch(err => {

    })
  }

  async loadModels() {
    this.model = await tf.loadLayersModel(environment.mainURL + '/models' + '/' + this.settings.MODEL_NAME + '/model.json')
    this.blaze = await blazeface.load({maxFaces:1, inputHeight: 128, inputWidth: 128})
    console.log(`loaded ${this.settings.MODEL_NAME}`)
  }

  bufftoarr(tensor: Tensor3D) {
    return tf.browser.toPixels(tensor)
  }

  /** makes prediction over 1 image */
  async predict(image: HTMLImageElement): Promise<tf.Tensor> {
    let i = this.loadImage(image)
    let face = await this.cropImage(i)
    let pred: tf.Tensor = this.model.predict(face.expandDims(), {batchSize: 1}) as tf.Tensor
    return pred
    // let pred: tf.Tensor | tf.Tensor[] = this.model.predict(tensor.expandDims(), {batchSize: 1})
    // return pred as tf.Tensor
  }

  /** convert htmlimage to tensor */
  loadImage(img: HTMLImageElement): tf.Tensor3D {
    let t = tf.browser.fromPixels(img)
    return t.pad(this.getPadding(t.shape))
  }

  /** returns face */
  async cropImage(tensor: Tensor3D): Promise<Tensor3D> {
    let t = tf.tidy(() => { return tensor.clone().resizeBilinear([256,256]) }) 
    let arr = await this.blaze.estimateFaces(t as Tensor3D, false)
    
    let tl: number[] = (arr[0].topLeft as [number, number]).map((e: number, i: number) => {return e/256}).reverse()
    let br: number[] = (arr[0].bottomRight as [number, number]).map((e: number, i: number) => {return e/256}).reverse()

    return tf.tidy(() => {
      return tf.image.cropAndResize(tensor.expandDims() as any, tf.tensor2d([tl.concat(br)]), [0], this.settings.IMG_SIZE).unstack()[0] as Tensor3D
    })
  }

  getPadding(shape: any): [number,number][] {
    let arr: [number,number][] = [[0,0],[0,0],[0,0]]
    let dif = shape[0]-shape[1]

    if (dif === 0 ) {
      return arr
    }

    dif>0? arr[1] = [Math.abs(dif/2), Math.abs(dif/2)] : arr[0] = [Math.abs(dif/2), Math.abs(dif/2)] 

    return arr
  }
}
