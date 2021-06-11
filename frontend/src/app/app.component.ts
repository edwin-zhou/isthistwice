import { TfserviceService } from './services/tfservice.service';
import { Component, OnInit } from '@angular/core';
import { TFSavedModel } from '@tensorflow/tfjs-node-gpu/dist/saved_model';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss']
})
export class AppComponent {
  title = 'frontend';

  constructor(private TfService: TfserviceService) {

  }

  OnInit() {

  }

  onFileChange(e: any) {
    let file = e.target.files[0]

    try {
      let t = this.TfService.loadImage(file)
    } catch (error) {
      
    }

  }

  onSubmit() {
  }
}
