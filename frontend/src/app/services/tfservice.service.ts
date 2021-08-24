import { environment } from 'src/environments/environment';
import { Injectable, OnInit, Output } from '@angular/core';
import * as tf from '@tensorflow/tfjs'
import * as blazeface from '@tensorflow-models/blazeface'
import { Tensor3D } from '@tensorflow/tfjs';
import { BehaviorSubject } from 'rxjs';
import { HttpClient, HttpParams } from '@angular/common/http';

@Injectable({
  providedIn: 'root'
})
export class TfserviceService {
  // model!: tf.LayersModel
  blaze!: blazeface.BlazeFaceModel

  settings: {
    [key: string]: any,
  } = {}
  
  private _modelLoaded: BehaviorSubject<boolean> = new BehaviorSubject(false as boolean)
  modelLoaded = this._modelLoaded.asObservable()

  constructor(private http: HttpClient) { 
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
    // this.model = await tf.loadLayersModel(environment.mainURL + '/models' + '/' + this.settings.MODEL_NAME + '/model.json')
    this.blaze = await blazeface.load({maxFaces:this.settings.LABELS.length, inputHeight: 128, inputWidth: 128})
  }

  async loadPred(t: tf.Tensor4D): Promise<any> {
    return this.http.post(environment.mainURL + '/models' + '/' + this.settings.MODEL_NAME, {
      image: t.arraySync()
    }).toPromise()
    .then((val: any) => {
      return val.pred
    })
    .catch(err => console.log(err))
  }

  bufftoarr(tensor: Tensor3D) {
    return tf.browser.toPixels(tensor)
  }

  /** makes prediction over 1 image */
  async predict(image: HTMLImageElement): Promise<{bbox: number[][], pred: number[][]}> {
    let i = this.loadImage(image)
    let faces: {t: tf.Tensor4D, bbox: number[][]} | void = await this.cropImage(i) 

    if (!faces) {
      let arr: number[] = new Array(this.settings.LABELS.length).fill(0, 0, this.settings.LABELS.length)
      tf.dispose(i)
      tf.dispose(faces)
      return {bbox: [arr], pred: [arr]} 
    }

    let pred: number[][] = await this.loadPred(faces.t)

    tf.dispose(i)
    tf.dispose(faces)
    return {bbox: faces.bbox, pred: pred} 
  }

  /** convert htmlimage to tensor */
  loadImage(img: HTMLImageElement): tf.Tensor3D {
    let t = tf.browser.fromPixels(img)
    return t.pad(this.getPadding(t.shape))
  }

  /** returns face */
  async cropImage(tensor: Tensor3D): Promise<{t: tf.Tensor4D, bbox: number[][]}|void> {
    let t = tf.tidy(() => { return tensor.clone().resizeBilinear([256,256]) }) 
    let bboxes = await this.blaze.estimateFaces(t as Tensor3D, false)

    if (!bboxes.length) {
      return
    }

    let faces: number[][] = []
    bboxes.forEach(face => {
      let tl: number[] = (face.topLeft as [number, number]).map((e: number, i: number) => {return e/256}).reverse()
      let br: number[] = (face.bottomRight as [number, number]).map((e: number, i: number) => {return e/256}).reverse() 
      faces.push(tl.concat(br))
    })

    return {
      t: tf.tidy(() => {
        return tf.image.cropAndResize(tensor.expandDims() as any, tf.tensor2d(faces), new Array(faces.length).fill(0), this.settings.IMG_SIZE)
        // return tf.image.cropAndResize(tensor.expandDims() as any, tf.tensor2d([face.tl.concat(face.br)]), [0], this.settings.IMG_SIZE).unstack()[0] as Tensor3D
      }),
      bbox: faces
    } 
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
