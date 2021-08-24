import { Tensor3D } from '@tensorflow/tfjs';
import { TfserviceService } from './services/tfservice.service';
import { Component, OnDestroy, OnInit } from '@angular/core';
import { FormControl, FormGroup, NgControl, Validators } from '@angular/forms';
import { Subscription } from 'rxjs';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss']
})
export class AppComponent implements OnInit, OnDestroy {
  sb: string = '&#127827'

  title = 'leappdepwinss';
  picBuff: any
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
  }
  
  deletePic() {
    this.ngOnDestroy()
    this.ngOnInit()
  }

  async onSubmit() {
    this.predicting = true
    if (this.picBuff) {
      let pic: HTMLImageElement = document.getElementById('subject') as HTMLImageElement
      let pred = await this.TfService.predict(pic)
      this.stats = pred[0]
      this.certainty = Math.max(...this.stats)
      this.prediction = this.certainty===0? "" : this.TfService.settings.LABELS[this.stats.indexOf(this.certainty)]
    }
    // else if (this.imageURL != '') {

    //   let pic = document.getElementById('urlsubject')
    //   let t = this.TfService.loadImage(pic)
    //   let pred: number[][] = this.TfService.predict(t).arraySync() as number[][]

    //   this.certainty = Math.max(...pred[0])
    //   this.prediction = this.TfService.settings.LABELS[pred[0].indexOf(this.certainty)]
    // }
    this.predicting = false
  }

  ngOnDestroy() {
    this.modelSub.unsubscribe()
  }

}
