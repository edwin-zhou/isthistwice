import { Tensor3D } from '@tensorflow/tfjs';
import { TfserviceService } from './services/tfservice.service';
import { Component, OnDestroy, OnInit } from '@angular/core';
import { FormControl, FormGroup, NgControl, Validators } from '@angular/forms';
import { Observable, Observer, Subscription } from 'rxjs';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss']
})
export class AppComponent implements OnInit, OnDestroy {
  sb: string = '&#127827'

  title = 'leappdepwinss'
  picBuff: any
  results: {bbox: number[][], pred: number[][]} = {bbox: [], pred: []}
  certainty: number = -1
  prediction: string = ''
  labels: string[] = []
  stats: number[] = []

  imageURL: string = ''

  form: FormGroup = new FormGroup({
    'pic': new FormControl(''),
    'url': new FormControl('')
  })

  modelSub!: Subscription
  modelLoaded: boolean = false
  predicting: boolean = false

  constructor(private TfService: TfserviceService) {
  }

  ngOnInit() {
    this.form.reset()
    this.picBuff = null
    this.imageURL = ''
    this.prediction = ''
    this.certainty = -1
    this.labels = []

    this.modelSub = this.TfService.modelLoaded.subscribe(o => {
      this.modelLoaded = o
      this.labels = this.TfService.settings.LABELS
    })
  }

  onFileChange(event: any) {
    const files = event.target.files;
    if (files.length === 0) {
      return;
    }

    const mimeType = files[0].type;
    if (mimeType.match(/image\/*/) == null) {
      return;
    } 

    const reader = new FileReader();
    reader.readAsDataURL(files[0]); 
    reader.onload = (_event) => { 
      this.picBuff = reader.result
      return
    }
  }

  onUrlChange(event: any) {
    this.imageURL = this.form.get('url')?.value
    this.getBase64ImageFromURL(this.imageURL).subscribe((base64data: any) => {    
      // this is the image as dataUrl
      let base64Image = 'data:image/jpg;base64,' + base64data;
    });
  }
  
  getBase64ImageFromURL(url: string) {
    return Observable.create((observer: Observer<string>) => {
      // create an image object
      let img = new Image();
      img.crossOrigin = 'Anonymous';
      img.src = url;
      if (!img.complete) {
          // This will call another method that will create image from url
          img.onload = () => {
          observer.next(this.getBase64Image(img));
          observer.complete();
        };
        img.onerror = (err) => {
           observer.error(err);
        };
      } else {
          observer.next(this.getBase64Image(img));
          observer.complete();
      }
    });
  }
  getBase64Image(img: HTMLImageElement) {
    // We create a HTML canvas object that will create a 2d image
    var canvas: HTMLCanvasElement = document.getElementById('canv') as HTMLCanvasElement
    canvas.width = img.width;
    canvas.height = img.height;
    var ctx = canvas.getContext("2d");
    // This will draw image    
    ctx!.drawImage(img, 0, 0);
    // Convert the drawn image to Data URL
    var dataURL = canvas.toDataURL("image/png");
    return dataURL.replace(/^data:image\/(png|jpg);base64,/, "");
  }
  
  deletePic() {
    this.ngOnDestroy()
    this.ngOnInit()
  }

  async onSubmit() {
    this.predicting = true
    if (this.picBuff) {
      let pic: HTMLImageElement = document.getElementById('subject') as HTMLImageElement
      this.results = await this.TfService.predict(pic)
      this.stats = this.results.pred[0]
      this.certainty = Math.max(...this.stats)
      this.prediction = this.certainty===0? "" : this.TfService.settings.LABELS[this.stats.indexOf(this.certainty)]
    }
    else if (this.imageURL != '') {
      this.results = await this.TfService.predict(document.getElementById('canv') as HTMLImageElement)
      this.stats = this.results.pred[0]
      this.certainty = Math.max(...this.stats)
      this.prediction = this.certainty===0? "" : this.TfService.settings.LABELS[this.stats.indexOf(this.certainty)]
    }
    this.predicting = false
  }

  ngOnDestroy() {
    this.modelSub.unsubscribe()
  }

}
