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

  constructor(private http: HttpClient) {}

  anonymize() {
    if (!this.inputText) {
      this.anonymizedText = '';
      return;
    }

    this.loading = true;
    this.errorMessage = '';

    this.http.post<{ anonymized: string }>(
      'https://your-backend-url/anonymize', // <-- replace with your actual API endpoint
      { text: this.inputText }
    ).pipe(
      catchError(err => {
        this.errorMessage = 'Failed to anonymize text. Try again.';
        this.loading = false;
        return throwError(() => err);
      })
    ).subscribe(response => {
      this.anonymizedText = response.anonymized;
      this.loading = false;
    });
  }

  deanonymize() {
    if (!this.inputText) {
      this.deanonymizedText = '';
      return;
    }

    this.loading = true;
    this.errorMessage = '';

    this.http.post<{ deanonymized: string }>(
      'https://your-backend-url/deanonymize', // <-- replace with your backend endpoint
      { text: this.inputText }
    ).pipe(
      catchError(err => {
        this.errorMessage = 'Failed to deanonymize text. Try again.';
        this.loading = false;
        return throwError(() => err);
      })
    ).subscribe(response => {
      this.deanonymizedText = response.deanonymized;
      this.loading = false;
    });
  }

  clear() {
    this.inputText = '';
    this.anonymizedText = '';
    this.deanonymizedText = '';
    this.errorMessage = '';
  }
}
