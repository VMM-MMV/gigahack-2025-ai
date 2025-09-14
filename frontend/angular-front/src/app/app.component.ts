import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { HttpClient, HttpClientModule } from '@angular/common/http';
import { catchError } from 'rxjs/operators';
import { throwError } from 'rxjs';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, FormsModule, HttpClientModule],
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent {
  title = 'Data Anonymization System';

  // Mode switching
  currentMode: 'anonymize' | 'deanonymize' = 'anonymize';
  setMode(mode: 'anonymize' | 'deanonymize') {
    this.currentMode = mode;
  }

  // Text bindings
  inputText = '';
  anonymizedText = '';
  deanonymizedText = '';
  loading = false;
  errorMessage = '';

  constructor(private http: HttpClient) { }

  metadata: any = null;
  isCollapsed = false; 

  anonymize() {
    this.http.post<any>('http://127.0.0.1:8000/anonymize', { text: this.inputText })
      .subscribe(response => {
        this.anonymizedText = response.anonymized_text;
        this.metadata = response.metadata; // save metadata for later
      });
  }

  deanonymize() {
    if (!this.anonymizedText || !this.metadata) {
      alert("Please anonymize first before deanonymizing.");
      return;
    }

    this.http.post<any>('http://127.0.0.1:8000/deanonymize', {
      text: this.anonymizedText,
      metadata: this.metadata
    }).subscribe(response => {
      this.deanonymizedText = response.deanonymized_text;
    });
  }




  clear() {
    this.inputText = '';
    this.anonymizedText = '';
    this.deanonymizedText = '';
    this.errorMessage = '';
  }
}
