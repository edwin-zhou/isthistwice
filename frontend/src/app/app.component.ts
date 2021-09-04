import { environment } from 'src/environments/environment';
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
      let img= new Image
      img.src = reader.result as string
      img.onload = () => {
        this.imageURL = img.src
        let canvas: HTMLCanvasElement = document.getElementById('canv') as HTMLCanvasElement
        let ctx = canvas.getContext("2d");
        canvas.width = img.width;
        canvas.height = img.height;
        ctx?.drawImage(img,0,0)
        return
      }
    }
  }

  onUrlChange(event: any) {
    this.imageURL = this.form.get('url')?.value
    this.getBase64ImageFromURL(this.imageURL).subscribe((base64data: any) => {    
      // this is the image as dataUrl
      // let base64Image = 'data:image/jpg;base64,' + base64data;
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

  getLabels(bboxes: number[][]) {
    let canvas: HTMLCanvasElement = document.getElementById('canv') as HTMLCanvasElement
    let ctx = canvas.getContext("2d");
    let d: number = Math.abs(canvas.width-canvas.height)

    bboxes.forEach((bbox: number[], index) => {
      let xTL: number = (canvas.width>canvas.height)? bbox[1]*canvas.width : bbox[1]*(canvas.width+d) - d/2 
      let yTL: number = (canvas.width>canvas.height)? bbox[0]*(canvas.height+d) - d/2 : bbox[0]*canvas.height
      let c: number = (canvas.width>canvas.height)? Math.abs(bbox[1]-bbox[3]) * canvas.width : Math.abs(bbox[0]-bbox[2]) * canvas.height

      let p =  environment.colorsRGB[this.results.pred[index].indexOf(Math.max(...this.results.pred[index]))]
      ctx!.beginPath();
      ctx!.rect(xTL, yTL, c, c);
      ctx!.strokeStyle = p
      ctx!.lineWidth = 15
      ctx!.stroke();
    })

  }
  
  deletePic() {
    this.ngOnDestroy()
    this.ngOnInit()
  }

  async onSubmit() {
    this.predicting = true
    this.results = await this.TfService.predict(document.getElementById('canv') as HTMLImageElement)
    this.stats = this.results.pred[0]
    this.certainty = Math.max(...this.stats)
    this.prediction = this.certainty===0? "" : this.TfService.settings.LABELS[this.stats.indexOf(this.certainty)]
    this.getLabels(this.results.bbox)

    this.predicting = false
  }

  ngOnDestroy() {
    this.modelSub.unsubscribe()
  }

}
