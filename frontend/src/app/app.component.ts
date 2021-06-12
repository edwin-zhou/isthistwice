import { TfserviceService } from './services/tfservice.service';
import { Component, OnInit } from '@angular/core';
import { FormControl, FormGroup, NgControl, Validators } from '@angular/forms';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss']
})
export class AppComponent implements OnInit {
  title = 'frontend';
  picBuff: any
  picURL: any

  form: FormGroup = new FormGroup({
    'pic': new FormControl('', [Validators.required])
  })

  constructor(private TfService: TfserviceService) {
  }

  ngOnInit() {
    this.form.reset()
    this.picBuff = null
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
      return null
    }
  }
  
  deletePic() {
    this.ngOnInit()
  }


  onSubmit() {
    if (this.picBuff) {
      console.log('submited')
    }
  }

}
